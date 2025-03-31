# -*- coding: utf-8 -*-
"""Automated Heatwave Prediction Model with Complete Implementation"""
import shutil
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "heatwave_data"
MODEL_DIR = "heatwave_models"
CITIES_FILE = "cities_to_train.csv"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class HeatwavePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['temperature_2m_max', 'apparent_temperature_max',
                        'relative_humidity_2m_mean', 'wind_speed_10m_max',
                        'pressure_msl_mean', 'precipitation_sum', 'cloud_cover_mean']
        self.data_file = os.path.join(DATA_DIR, "all_cities_data.csv")
        self.model_file = os.path.join(MODEL_DIR, "heatwave_model.pkl")
        self.scaler_file = os.path.join(MODEL_DIR, "heatwave_scaler.pkl")
        self.city_coordinates = {}
        self.geolocator = Nominatim(user_agent="heatwave_predictor")
        self.alert_thresholds = {
            'Normal': 0.3,
            'Caution': 0.6,
            'Warning': 0.8,
            'Emergency': 1.0
        }

    def get_coordinates(self, city_name):
        """Convert city name to coordinates with caching"""
        if city_name in self.city_coordinates:
            return self.city_coordinates[city_name]
            
        try:
            location = self.geolocator.geocode(city_name)
            if location:
                self.city_coordinates[city_name] = (location.latitude, location.longitude)
                return location.latitude, location.longitude
            else:
                raise ValueError(f"Coordinates not found for: {city_name}")
        except Exception as e:
            raise ValueError(f"Geocoding error for {city_name}: {str(e)}")

    def load_cities_to_train(self):
        """Load list of cities from CSV file"""
        if not os.path.exists(CITIES_FILE):
            raise FileNotFoundError(f"Cities file not found: {CITIES_FILE}")
        
        cities_df = pd.read_csv(CITIES_FILE)
        if 'city' not in cities_df.columns:
            raise ValueError("Cities CSV must have a 'city' column")
        
        # Set default threshold if not specified
        if 'threshold_temp' not in cities_df.columns:
            cities_df['threshold_temp'] = None
            
        return cities_df[['city', 'threshold_temp']].to_dict('records')

    def clean_data(self, df):
        """Handle missing values and data quality issues"""
        if 'is_heatwave' in df.columns:
            df = df.dropna(subset=['is_heatwave'])
        
        for feature in self.features:
            if feature in df.columns:
                if feature == 'precipitation_sum':
                    df[feature] = df[feature].fillna(0)
                elif feature.endswith('_mean') or feature.endswith('_max'):
                    df[feature] = df[feature].fillna(df[feature].rolling(3, min_periods=1).mean())
        
        df = df.dropna(subset=self.features)
        return df

    def determine_threshold(self, city_data, specified_threshold=None):
        """Determine the heatwave threshold temperature"""
        if specified_threshold is not None:
            return float(specified_threshold)
            
        abs_threshold = city_data['temperature_2m_max'].quantile(0.95)
        season_avg = city_data['temperature_2m_max'].mean()
        season_std = city_data['temperature_2m_max'].std()
        rel_threshold = season_avg + 1.5 * season_std
        
        final_threshold = max(abs_threshold, rel_threshold)
        print(f"Automatically determined threshold: "
              f"95th percentile={abs_threshold:.1f}°C, "
              f"relative={rel_threshold:.1f}°C → Using {final_threshold:.1f}°C")
        
        return final_threshold

    def fetch_historical_data(self, city_info, years=10):
        """Fetch historical weather data for a city with better error handling"""
        city = city_info['city']
        threshold_temp = city_info.get('threshold_temp')
        
        try:
            lat, lon = self.get_coordinates(city)
        except Exception as e:
            print(f"Failed to get coordinates for {city}: {str(e)}")
            raise

        # Check for existing data with better validation
        if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
            try:
                existing_data = pd.read_csv(self.data_file)
                if not existing_data.empty:
                    city_data = existing_data[existing_data['city'] == city]
                    if not city_data.empty:
                        print(f"Using existing data for {city}")
                        return city_data.copy()
            except Exception as e:
                print(f"Warning: Error reading existing data file: {str(e)}")

        print(f"Fetching {years} years of Feb-May data for {city}...")
        current_date = datetime.now().date()
        all_data = []

        for year_offset in range(1, years + 1):
            year = current_date.year - year_offset
            start_date = datetime(year, 2, 1).date()  # Feb 1st
            end_date = datetime(year, 5, 31).date()   # May 31st
            
            if year == current_date.year:
                end_date = min(end_date, current_date - timedelta(days=1))
                if end_date < start_date:
                    continue

            try:
                # Skip invalid date ranges
                if start_date > end_date:
                    print(f"Skipping {year} (invalid date range)")
                    continue

                response = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "daily": self.features,
                        "timezone": "auto"
                    },
                    timeout=50
                )
            
                response.raise_for_status()
                data = response.json()
                
                # Validate API response
                if not data.get("daily"):
                    print(f"No daily data in response for {city} ({year})")
                    continue
                    
                df_year = pd.DataFrame(data["daily"])
                if df_year.empty:
                    print(f"Empty dataframe for {city} ({year})")
                    continue
                    
                df_year['city'] = city
                df_year['year'] = year
                all_data.append(df_year)
                print(f"Fetching Feb-May data for {year}...")
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed for {city} ({year}): {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing data for {city} ({year}): {str(e)}")
                continue

        if not all_data:
                raise ValueError(f"Failed to fetch any historical data for {city}")
            
        try:
            df = pd.concat(all_data).drop_duplicates(["time", "city"])
            df['time'] = pd.to_datetime(df['time'])

            # Calculate rolling averages
            df['rolling_temp'] = df.groupby('city')['temperature_2m_max'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

            # Determine threshold
            final_threshold = self.determine_threshold(df, threshold_temp)
            
            # Calculate season stats
            city_stats = df.groupby('city')['temperature_2m_max'].agg(['mean', 'std']).reset_index()
            city_stats.columns = ['city', 'season_avg', 'season_std']
            df = pd.merge(df, city_stats, on='city')

            # Label heatwaves
            relative_threshold = df['season_avg'] + 1.5 * df['season_std']
            df['is_heatwave'] = 0
            
            # Apply both absolute and relative thresholds
            df.loc[df['temperature_2m_max'] > final_threshold, 'is_heatwave'] = 1
            
            for city_name in df['city'].unique():
                city_mask = df['city'] == city_name
                city_df = df[city_mask].copy()
                rel_threshold = city_df['season_avg'].iloc[0] + 1.5 * city_df['season_std'].iloc[0]
                
                for i in range(len(city_df) - 2):
                    if all(city_df.iloc[i:i+3]['temperature_2m_max'] > rel_threshold):
                        idx = city_df.index[i:i+3]
                        df.loc[idx, 'is_heatwave'] = 1

            # Clean and save data
            df = self.clean_data(df)
            try:
                if os.path.exists(self.data_file):
                    existing_data = pd.read_csv(self.data_file)
                    combined_data = pd.concat([existing_data, df]).drop_duplicates(["time", "city"])
                    combined_data.to_csv(self.data_file, index=False)
                else:
                    df.to_csv(self.data_file, index=False)
            except Exception as e:
                print(f"Warning: Failed to save data: {str(e)}")
                raise
                
            return df
            
        except Exception as e:
            print(f"Error processing data for {city}: {str(e)}")
            raise

    def prepare_features(self, df):
        """Prepare features for model training"""
        df = self.clean_data(df)
        X = df[self.features].copy()
        y = df['is_heatwave']

        X['temp_humidity_ratio'] = X['temperature_2m_max'] / (X['relative_humidity_2m_mean'] + 0.001)  # Avoid division by zero
        X['diurnal_range'] = X['temperature_2m_max'] - X['apparent_temperature_max']

        assert not X.isnull().values.any(), "NaN values detected in features"
        X_scaled = self.scaler.fit_transform(X)
    
        return X_scaled, y

    def train_on_all_cities(self, test_size=0.2, random_state=42, years=10):
        """Train model on all cities from the cities file"""
        cities_info = self.load_cities_to_train()
        all_data = pd.DataFrame()

        for city_info in cities_info:
            try:
                city_data = self.fetch_historical_data(city_info, years=years) # type: ignore # Pass years parameter
                all_data = pd.concat([all_data, city_data])
                print(f"Successfully processed data for {city_info['city']}")
            except Exception as e:
                print(f"Failed to process {city_info['city']}: {str(e)}")

        if all_data.empty:
            raise ValueError("No valid data available for training")

        # Prepare features
        X, y = self.prepare_features(all_data)

        if len(y.unique()) < 2:
            raise ValueError("Only one class present in the data. Adjust the heatwave thresholds.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Calculate sample weights
        class_weights = {0: 1, 1: len(y_train)/sum(y_train)}
        
        models = {
            'RandomForest': RandomForestClassifier(class_weight=class_weights, random_state=random_state),
            'GradientBoosting': GradientBoostingClassifier(random_state=random_state)
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            print(f"\nTraining {name} model...")
            if name == 'GradientBoosting':
                sample_weight = np.where(y_train==1, class_weights[1], class_weights[0])
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)

            param_grid = {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }[name]

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            print(f"Best parameters: {grid_search.best_params_}")
            best_model_cv = grid_search.best_estimator_
            y_pred = best_model_cv.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            if f1 > best_score:
                best_score = f1
                best_model = best_model_cv

        # Save the best model
        self.model = best_model
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\nModel trained on {len(cities_info)} cities saved to {self.model_file}")
        print(f"Best model F1 score: {best_score:.4f}")

        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.features + ['temp_humidity_ratio', 'diurnal_range']
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importance)), importance[indices], align='center')
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

        return self.model

    def load_model(self):
        """Load the saved model"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Model loaded from {self.model_file}")
            return True
        else:
            print("No saved model found")
            return False
        
    def fetch_forecast_data(self, city):
        """Fetch 7-day forecast data for prediction"""
        try:
            lat, lon = self.get_coordinates(city)
            
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": self.features,
                    "forecast_days": 7,
                    "timezone": "auto"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            forecast_df = pd.DataFrame(data["daily"])
            forecast_df['city'] = city
            return forecast_df
            
        except Exception as e:
            raise ValueError(f"Failed to fetch forecast for {city}: {str(e)}")

    def predict_heatwave(self, city):
        """Predict heatwave with better-calibrated probabilities"""
        
        if self.model is None and not self.load_model():
            raise ValueError("No trained model available")
        
        try:
            # Get forecast data
            forecast_df = self.fetch_forecast_data(city)
            
            # Clean and prepare data
            forecast_df = self.clean_data(forecast_df)
            X = forecast_df[self.features].copy()
            
            # Add derived features
            X['temp_humidity_ratio'] = X['temperature_2m_max'] / (X['relative_humidity_2m_mean'].clip(lower=1))
            X['diurnal_range'] = X['temperature_2m_max'] - X['apparent_temperature_max']
            
            # Get base probabilities
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # 1. Climate-adjusted calibration
            if os.path.exists(self.data_file):
                city_data = pd.read_csv(self.data_file)
                if city in city_data['city'].unique():
                    city_stats = city_data[city_data['city'] == city]['temperature_2m_max'].describe()
                    # Normalize probabilities based on city's historical temps
                    temp_norm = (forecast_df['temperature_2m_max'] - city_stats['mean']) / city_stats['std']
                    probabilities = 1 / (1 + np.exp(-(temp_norm * 0.5)))  # More gradual curve
            
            # 2. Temperature-based adjustment (more conservative)
            temp_boost = np.clip(
                (forecast_df['temperature_2m_max'] - 35) / 15,  # More gradual increase
                0, 0.3  # Max boost of 30%
            )
            probabilities = np.clip(probabilities + temp_boost, 0.01, 0.99)  # Never 0% or 100%
            
            # 3. Consecutive day effect (smoothed)
            for i in range(1, len(probabilities)):
                if probabilities[i - 1] > 0.3:
                    # Gradual increase that diminishes over time
                    probabilities[i] = np.clip(probabilities[i] * (1 + 0.1 * (probabilities[i - 1] - 0.3) / 0.7), 0, 0.95)
            
            # 4. Humidity modulation
            humidity_effect = 1 - (X['relative_humidity_2m_mean'] / 100) ** 2  # More impact at low humidity
            probabilities = probabilities * (0.7 + 0.3 * humidity_effect)
            
            # Add predictions to dataframe
            forecast_df['is_heatwave'] = (probabilities > 0.5).astype(int)
            forecast_df['heatwave_probability'] = probabilities

            """Predict heatwave with classified alerts"""
            
            # Type-safe alert classification
            conditions = [
                (probabilities <= 0.3),
                (probabilities <= 0.6),
                (probabilities <= 0.8),
                (probabilities > 0.8)
            ]
            
            # Convert all to string type explicitly
            alerts = np.array(['Normal', 'Caution', 'Warning', 'Emergency'], dtype='object')
            colors = np.array(['green', 'yellow', 'orange', 'red'], dtype='object')
            actions = np.array([
                "No special precautions needed",
                "Stay hydrated, limit outdoor activities",
                "Avoid outdoor activities 11am-4pm",
                "Extreme danger - stay indoors"
            ], dtype='object')
            
            forecast_df['alert_level'] = np.select(conditions, alerts, default='Unknown')
            forecast_df['alert_color'] = np.select(conditions, colors, default='gray')
            forecast_df['recommended_action'] = np.select(conditions, actions, default='Monitor conditions')
            
            return forecast_df
            
        except Exception as e:
            raise ValueError(f"Prediction failed for {city}: {str(e)}")

    def plot_probability_calibration(y_true, probs):
        """Visual check of probability reliability"""
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10)
        
        plt.figure(figsize=(10, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Actual Fraction Positive")
        plt.title("Probability Calibration Curve")
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_forecast(self, forecast_df):
        """Visualize the forecast with heatwave predictions"""
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        city = forecast_df['city'].iloc[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Temperature plot
        ax1.plot(forecast_df['time'], forecast_df['temperature_2m_max'], 'ro-', label='Max Temp (°C)')
        ax1.plot(forecast_df['time'], forecast_df['apparent_temperature_max'], 'bo-', label='Feels Like (°C)')

        for i, row in forecast_df.iterrows():
            if row['is_heatwave'] == 1:
                ax1.axvspan(row['time'] - pd.Timedelta(hours=12), row['time'] + pd.Timedelta(hours=12),
                           alpha=0.3, color='red')

        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'Heatwave Forecast for {city}')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Probability plot
        ax2.bar(forecast_df['time'], forecast_df['heatwave_probability'], color='orange', alpha=0.7)
        ax2.axhline(y=0.5, color='r', linestyle='--')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Date')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\nHeatwave Forecast for {city}:")
        print("--------------------------")
        for i, row in forecast_df.iterrows():
            date_str = row['time'].strftime('%Y-%m-%d')
            prob = row['heatwave_probability'] * 100
            status = "HEATWAVE ALERT 🔥" if row['is_heatwave'] == 1 else "Normal"
            print(f"{date_str}: Temp: {row['temperature_2m_max']}°C, Prob: {prob:.1f}% - {status}")
        
        """Visualize forecast with alert classification"""
        plt.figure(figsize=(14, 10))
        
        # Create alert color mapping
        alert_colors = forecast_df['alert_color'].tolist()
        
        # Plot temperature with alert background
        for i, row in forecast_df.iterrows():
            plt.axvspan(i-0.5, i+0.5, color=row['alert_color'], alpha=0.2)
        
        plt.plot(forecast_df['temperature_2m_max'], 'ro-', label='Max Temp (°C)')
        
        # Add alert annotations
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            plt.text(i, row['temperature_2m_max']+0.5, 
                    row['alert_level'], 
                    color=row['alert_color'],
                    ha='center')
        
        plt.title(f"Heatwave Alert Forecast for {forecast_df['city'].iloc[0]}")
        plt.ylabel("Temperature (°C)")
        plt.xticks(range(len(forecast_df)), forecast_df['time'].dt.strftime('%a %m/%d'))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print detailed alert summary
        print("\n🚨 Heatwave Alert Summary:")
        print("--------------------------")
        for _, row in forecast_df.iterrows():
            print(f"{row['time'].strftime('%Y-%m-%d')}: "
                f"{row['alert_level']} ({row['heatwave_probability']*100:.0f}%) - "
                f"{row['recommended_action']}")

def main():
    model = HeatwavePredictionModel()
    
    print("Starting automated training process...")
    try:
        # 1. Train the model
        model.train_on_all_cities(years=5)
        print("\nTraining completed successfully!")
        
        # 2. Export model files to prediction package
        os.makedirs("models_files", exist_ok=True)  # Create dir if needed
        try:    
            shutil.copy("heatwave_models/heatwave_model.pkl", "models_files/")
            shutil.copy("heatwave_models/heatwave_scaler.pkl", "models_files/")
            print("Model files exported to prediction_package directory")
        except IOError as e:
            print(f"Failed to export model: {str(e)}")    

        # 3. Continue with prediction interface
        while True:
            city = input("\nEnter city to predict (or 'quit' to exit): ")
            if city.lower() == 'quit':
                break
                
            try:
                forecast = model.predict_heatwave(city)
                model.visualize_forecast(forecast)
            except Exception as e:
                print(f"Prediction failed: {str(e)}")
                
    except Exception as e:
        print(f"\nError during training: {str(e)}")

if __name__ == "__main__":
    main()