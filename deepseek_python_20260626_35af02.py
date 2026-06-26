"""
REAL ESTATE PRICE PREDICTION REST API
Complete FastAPI implementation with model training and API endpoints
Run this file directly - no external data files needed!
"""

import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

# =============================================
# STEP 1: TRAIN THE MODEL (if not already saved)
# =============================================

def train_and_save_model():
    """Train the property price prediction model and save it"""
    
    print("🏠 Training property price prediction model...")
    
    # Import required libraries for training
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    
    np.random.seed(42)
    n_samples = 2000
    
    # Generate synthetic data
    square_footage = np.random.uniform(800, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4.5, n_samples).round(1)
    year_built = np.random.randint(1950, 2024, n_samples)
    property_age = 2024 - year_built
    
    locations = np.random.choice(
        ['Downtown', 'Suburb', 'Rural', 'Urban', 'Waterfront'],
        n_samples,
        p=[0.25, 0.30, 0.15, 0.20, 0.10]
    )
    
    has_pool = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    has_garage = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    has_basement = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    has_central_air = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Calculate prices
    location_multipliers = {
        'Downtown': 450,
        'Urban': 380,
        'Suburb': 320,
        'Waterfront': 520,
        'Rural': 250
    }
    
    base_price = np.zeros(n_samples)
    for i, loc in enumerate(locations):
        base_price[i] = square_footage[i] * location_multipliers[loc]
    
    bedroom_value = bedrooms * 15000
    bathroom_value = bathrooms * 12000
    depreciation = np.maximum(0, property_age * 0.005)
    
    amenity_bonus = (
        has_pool * 25000 +
        has_garage * 15000 +
        has_basement * 20000 +
        has_central_air * 10000
    )
    
    pool_waterfront_bonus = has_pool * (locations == 'Waterfront') * 30000
    noise = np.random.normal(0, 30000, n_samples)
    
    price = (
        base_price * (1 - depreciation) +
        bedroom_value +
        bathroom_value +
        amenity_bonus +
        pool_waterfront_bonus +
        noise
    )
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'property_age': property_age,
        'location': locations,
        'has_pool': has_pool,
        'has_garage': has_garage,
        'has_basement': has_basement,
        'has_central_air': has_central_air,
        'price': price
    })
    
    # Feature engineering
    data['sqft_per_bedroom'] = data['square_footage'] / data['bedrooms']
    data['bath_bed_ratio'] = data['bathrooms'] / data['bedrooms']
    data['age_squared'] = data['property_age'] ** 2
    data['is_modern'] = (data['year_built'] >= 2000).astype(int)
    
    # Prepare features
    feature_cols = [
        'square_footage', 'bedrooms', 'bathrooms', 'property_age',
        'sqft_per_bedroom', 'bath_bed_ratio', 'age_squared', 'is_modern',
        'location', 'has_pool', 'has_garage', 'has_basement', 'has_central_air'
    ]
    
    X = data[feature_cols]
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessing pipeline
    categorical_cols = ['location']
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])
    
    # Create and train XGBoost model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ Model trained successfully!")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:,.2f}")
    
    # Save model and feature columns
    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'r2_score': r2,
        'rmse': rmse,
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'property_price_model.pkl')
    print("💾 Model saved as 'property_price_model.pkl'")
    
    return model_data

# Check if model exists, if not train it
if not os.path.exists('property_price_model.pkl'):
    print("📦 Model not found. Training new model...")
    model_data = train_and_save_model()
else:
    print("📂 Loading existing model...")
    model_data = joblib.load('property_price_model.pkl')
    print(f"✅ Model loaded successfully!")
    print(f"   R² Score: {model_data['r2_score']:.4f}")
    print(f"   Training date: {model_data['training_date']}")

# =============================================
# STEP 2: DEFINE FASTAPI APP
# =============================================

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API for predicting property prices based on property features",
    version="1.0.0"
)

# =============================================
# STEP 3: DEFINE REQUEST/RESPONSE MODELS
# =============================================

class PropertyFeatures(BaseModel):
    """Request model for property price prediction"""
    
    square_footage: float = Field(
        ..., 
        description="Total square footage of the property",
        ge=300, 
        le=10000,
        example=2500
    )
    bedrooms: int = Field(
        ..., 
        description="Number of bedrooms",
        ge=0, 
        le=10,
        example=3
    )
    bathrooms: float = Field(
        ..., 
        description="Number of bathrooms (can be fractional)",
        ge=0, 
        le=8,
        example=2.5
    )
    year_built: int = Field(
        ..., 
        description="Year the property was built",
        ge=1800, 
        le=2024,
        example=2010
    )
    location: str = Field(
        ..., 
        description="Property location",
        example="Downtown"
    )
    has_pool: bool = Field(
        False,
        description="Whether the property has a pool",
        example=True
    )
    has_garage: bool = Field(
        False,
        description="Whether the property has a garage",
        example=True
    )
    has_basement: bool = Field(
        False,
        description="Whether the property has a basement",
        example=False
    )
    has_central_air: bool = Field(
        False,
        description="Whether the property has central air conditioning",
        example=True
    )
    
    @validator('location')
    def validate_location(cls, v):
        valid_locations = ['Downtown', 'Suburb', 'Rural', 'Urban', 'Waterfront']
        if v not in valid_locations:
            raise ValueError(f"Location must be one of: {valid_locations}")
        return v
    
    @validator('year_built')
    def validate_year_built(cls, v):
        if v > 2024:
            raise ValueError("Year built cannot be in the future")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "square_footage": 2500,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "year_built": 2010,
                "location": "Downtown",
                "has_pool": True,
                "has_garage": True,
                "has_basement": False,
                "has_central_air": True
            }
        }

class PricePredictionResponse(BaseModel):
    """Response model for price prediction"""
    
    predicted_price: float = Field(
        ..., 
        description="Predicted property price in USD",
        example=750000
    )
    predicted_price_formatted: str = Field(
        ...,
        description="Formatted price with currency symbol",
        example="$750,000.00"
    )
    features: PropertyFeatures = Field(
        ...,
        description="Input features used for prediction"
    )
    model_metrics: dict = Field(
        ...,
        description="Model performance metrics"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )
    timestamp: str = Field(
        ...,
        description="Prediction timestamp"
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    properties: List[PropertyFeatures] = Field(
        ...,
        description="List of properties to predict prices for",
        min_items=1,
        max_items=100
    )
    
    class Config:
        schema_extra = {
            "example": {
                "properties": [
                    {
                        "square_footage": 2500,
                        "bedrooms": 3,
                        "bathrooms": 2.5,
                        "year_built": 2010,
                        "location": "Downtown",
                        "has_pool": True,
                        "has_garage": True,
                        "has_basement": False,
                        "has_central_air": True
                    },
                    {
                        "square_footage": 1800,
                        "bedrooms": 2,
                        "bathrooms": 2.0,
                        "year_built": 2005,
                        "location": "Suburb",
                        "has_pool": False,
                        "has_garage": True,
                        "has_basement": True,
                        "has_central_air": True
                    }
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    predictions: List[PricePredictionResponse] = Field(
        ...,
        description="List of predictions for each property"
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    average_price: float = Field(
        ...,
        description="Average predicted price across all properties"
    )
    max_price: float = Field(
        ...,
        description="Maximum predicted price"
    )
    min_price: float = Field(
        ...,
        description="Minimum predicted price"
    )

class HealthCheckResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_metrics: dict = Field(..., description="Model performance metrics")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")

# =============================================
# STEP 4: HELPER FUNCTIONS
# =============================================

import uuid
from datetime import datetime

def predict_price(features: dict) -> float:
    """Make price prediction using the loaded model"""
    
    # Convert features to DataFrame
    df = pd.DataFrame([features])
    
    # Calculate derived features
    df['property_age'] = 2024 - df['year_built']
    df['sqft_per_bedroom'] = df['square_footage'] / df['bedrooms']
    df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms']
    df['age_squared'] = df['property_age'] ** 2
    df['is_modern'] = (df['year_built'] >= 2000).astype(int)
    
    # Get model
    model = model_data['model']
    
    # Make prediction
    prediction = model.predict(df[model_data['feature_columns']])[0]
    
    return prediction

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

# =============================================
# STEP 5: API ENDPOINTS
# =============================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Real Estate Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict price for a single property",
            "/predict/batch": "POST - Predict prices for multiple properties",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - API documentation (ReDoc)"
        }
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if model_data['model'] is not None else "unhealthy",
        model_loaded=model_data['model'] is not None,
        model_metrics={
            "r2_score": model_data.get('r2_score', None),
            "rmse": model_data.get('rmse', None)
        },
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PricePredictionResponse)
async def predict_property_price(property: PropertyFeatures):
    """
    Predict price for a single property
    
    - **square_footage**: Total square footage (300-10,000 sq ft)
    - **bedrooms**: Number of bedrooms (0-10)
    - **bathrooms**: Number of bathrooms (0-8)
    - **year_built**: Year the property was built (1800-2024)
    - **location**: Property location (Downtown, Suburb, Rural, Urban, Waterfront)
    - **has_pool**: Whether property has a pool
    - **has_garage**: Whether property has a garage
    - **has_basement**: Whether property has a basement
    - **has_central_air**: Whether property has central air conditioning
    """
    try:
        # Convert to dict
        features = property.dict()
        
        # Make prediction
        predicted_price = predict_price(features)
        
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Create response
        response = PricePredictionResponse(
            predicted_price=predicted_price,
            predicted_price_formatted=format_currency(predicted_price),
            features=property,
            model_metrics={
                "r2_score": model_data.get('r2_score', None),
                "rmse": model_data.get('rmse', None),
                "training_date": model_data.get('training_date', None)
            },
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_properties(batch_request: BatchPredictionRequest):
    """
    Predict prices for multiple properties in a single request
    
    - **properties**: List of property feature objects (1-100 properties)
    """
    try:
        predictions = []
        prices = []
        
        for property_features in batch_request.properties:
            # Make prediction
            features = property_features.dict()
            predicted_price = predict_price(features)
            prices.append(predicted_price)
            
            # Create individual response
            pred_response = PricePredictionResponse(
                predicted_price=predicted_price,
                predicted_price_formatted=format_currency(predicted_price),
                features=property_features,
                model_metrics={
                    "r2_score": model_data.get('r2_score', None),
                    "rmse": model_data.get('rmse', None),
                    "training_date": model_data.get('training_date', None)
                },
                request_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat()
            )
            predictions.append(pred_response)
        
        # Aggregate statistics
        avg_price = np.mean(prices)
        max_price = np.max(prices)
        min_price = np.min(prices)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            average_price=avg_price,
            max_price=max_price,
            min_price=min_price
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/predict/features", response_model=dict)
async def predict_with_custom_features(features: dict):
    """
    Predict price with custom feature dictionary (flexible input)
    
    This endpoint accepts a flexible JSON structure and attempts to extract
    the required features for prediction.
    """
    try:
        # Validate required features
        required_features = ['square_footage', 'bedrooms', 'bathrooms', 
                           'year_built', 'location']
        
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Set default values for optional features
        optional_features = {
            'has_pool': False,
            'has_garage': False,
            'has_basement': False,
            'has_central_air': False
        }
        
        for key, default in optional_features.items():
            if key not in features:
                features[key] = default
        
        # Make prediction
        predicted_price = predict_price(features)
        
        return {
            "predicted_price": predicted_price,
            "predicted_price_formatted": format_currency(predicted_price),
            "features_used": features,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": model_data['model'] is not None,
        "feature_columns": model_data.get('feature_columns', []),
        "categorical_features": model_data.get('categorical_cols', []),
        "numeric_features": model_data.get('numeric_cols', []),
        "performance_metrics": {
            "r2_score": model_data.get('r2_score', None),
            "rmse": model_data.get('rmse', None)
        },
        "training_date": model_data.get('training_date', None),
        "prediction_endpoint": "/predict",
        "batch_endpoint": "/predict/batch"
    }

# =============================================
# STEP 6: ERROR HANDLING
# =============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# =============================================
# STEP 7: CORS MIDDLEWARE (Optional)
# =============================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# STEP 8: RUN THE API
# =============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING REAL ESTATE PRICE PREDICTION API")
    print("="*60)
    print("\n📋 API Documentation:")
    print("   Swagger UI: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print("   Health Check: http://localhost:8000/health")
    print("\n🔮 Test Prediction:")
    print("   curl -X POST http://localhost:8000/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"square_footage\": 2500, \"bedrooms\": 3, ...}'")
    print("\n" + "="*60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )