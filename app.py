from flask import Flask, request, jsonify
from flask_cors import CORS
from heatwave_model import HeatwavePredictionModel
import os
import json
import pandas as pd

app = Flask(__name__)
CORS(app, origins=["https://resplendent-macaron-7061c7.netlify.app"])

# Initialize model
model = HeatwavePredictionModel()
model.model_file = os.path.join('model_files', 'heatwave_model.pkl')
model.scaler_file = os.path.join('model_files', 'heatwave_scaler.pkl')

@app.route('/heatwave', methods=['POST'])
def predict_heatwave():
    try:
        # First check if request has data
        if not request.data:
            return jsonify({
                "error": "Empty request body",
                "message": "Request must contain JSON data",
                "status": "error"
            }), 400
            
        # Try to parse JSON
        try:
            data = request.get_json()
        except json.JSONDecodeError:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request must be valid JSON",
                "status": "error"
            }), 400
            
        if not data or 'city' not in data:
            return jsonify({
                "error": "Missing parameter",
                "message": "City parameter is required in JSON body",
                "status": "error"
            }), 400
        
        city = data['city']
        
        # Get predictions using your model
        forecast = model.predict_heatwave(city)
        
        # Convert forecast to proper format
        predictions = []
        for _, row in forecast.iterrows():
            # Handle different date types (Timestamp, datetime, string)
            date_value = row['time']
            
            # Convert to datetime if not already
            if isinstance(date_value, str):
                date_obj = pd.to_datetime(date_value, errors='coerce')
            elif hasattr(date_value, 'strftime'):
                date_obj = date_value
            else:
                date_obj = pd.to_datetime(date_value, unit='ns')  # Handle numpy datetime64
            
            formatted_date = date_obj.strftime('%Y-%m-%d') if not pd.isnull(date_obj) else "N/A"
            predictions.append({
                "date": formatted_date,
                "temperature_2m_max": row['temperature_2m_max'],
                "apparent_temperature_max": row['apparent_temperature_max'],
                "relative_humidity_2m_mean": row['relative_humidity_2m_mean'],
                "wind_speed_10m_max": row['wind_speed_10m_max'],
                "cloud_cover_mean": row['cloud_cover_mean'],
                "precipitation_sum": row['precipitation_sum'],
                "is_heatwave": int(row['is_heatwave']),
                "heatwave_probability": float(row['heatwave_probability']),
                "alert_level": row['alert_level'],
                "alert_color": row['alert_color'],
                "recommended_action": row['recommended_action']
            })
        
        return jsonify({
            "city": city,
            "predictions": predictions,
            "message": "Forecast generated successfully",
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to process your request",
            "status": "error"
        }), 500

@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    try:
        city = request.args.get('city')
        
        if not city:
            return jsonify({
                "error": "Missing parameter",
                "message": "City parameter is required",
                "status": "error"
            }), 400
        
        # Use your model's historical data fetching capability
        if not hasattr(model, 'data_file') or not os.path.exists(model.data_file):
            return jsonify({
                "error": "Data not available",
                "message": "Historical data file not found",
                "status": "error"
            }), 404
        
        # Load historical data from your model's data file
        historical_data = pd.read_csv(model.data_file)
        city_data = historical_data[historical_data['city'] == city]
        
        if city_data.empty:
            return jsonify({
                "error": "Data not found",
                "message": f"No historical data available for {city}",
                "status": "error"
            }), 404
        
        # Process and format the data
        formatted_data = []
        for year in city_data['year'].unique():
            year_data = city_data[city_data['year'] == year]
            formatted_data.append({
                "year": int(year),
                "max_temp": float(year_data['temperature_2m_max'].max()),
                "min_temp": float(year_data['temperature_2m_max'].min()),
                "avg_temp": float(year_data['temperature_2m_max'].mean()),
                "heatwave_days": int(year_data['is_heatwave'].sum()),
                "precipitation": float(year_data['precipitation_sum'].sum()),
                "humidity": float(year_data['relative_humidity_2m_mean'].mean())
            })
        
        return jsonify({
            "city": city,
            "historical_data": formatted_data,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to fetch historical data",
            "status": "error"
        }), 500

if __name__ == '__main__':
    # Ensure the model is loaded
    if not model.load_model():
        print("Warning: Could not load model files. Some functionality may be limited.")
    app.run(host='0.0.0.0', port=10000, debug=True)