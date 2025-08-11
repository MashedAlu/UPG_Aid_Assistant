import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, date
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Ultra-Poor Aid Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# SAINT Model Classes (same as your training script)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.multihead_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SAINT(nn.Module):
    def __init__(self, 
                 num_numerical_features,
                 categorical_features_info,
                 d_model=128,
                 n_heads=8,
                 n_layers=6,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()
        
        self.num_numerical_features = num_numerical_features
        self.categorical_features_info = categorical_features_info
        self.d_model = d_model
        
        self.numerical_projection = nn.Linear(1, d_model)
        
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, d_model) 
            for _, num_categories in categorical_features_info
        ])
        
        max_features = num_numerical_features + len(categorical_features_info)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_features, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * max_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, numerical_features, categorical_features):
        batch_size = numerical_features.size(0)
        
        numerical_tokens = []
        for i in range(self.num_numerical_features):
            token = self.numerical_projection(numerical_features[:, i:i+1])
            numerical_tokens.append(token.unsqueeze(1))
        
        categorical_tokens = []
        for i, embedding_layer in enumerate(self.categorical_embeddings):
            cat_emb = embedding_layer(categorical_features[:, i]).unsqueeze(1)
            categorical_tokens.append(cat_emb)
        
        all_tokens = numerical_tokens + categorical_tokens
        all_embeddings = torch.cat(all_tokens, dim=1)
        
        seq_len = all_embeddings.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        all_embeddings = all_embeddings + pos_encoding
        
        x = all_embeddings
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        x = x.view(batch_size, -1)
        output = self.classifier(x)
        
        return output

@st.cache_resource
def load_model():
    """Load the trained SAINT model"""
    try:
        # Load model checkpoint
        checkpoint = torch.load('final_saint_aid_prediction_model.pth', map_location='cpu', weights_only=False)
        
        # Extract model components
        scaler = checkpoint['scaler']
        categorical_features_info = checkpoint['categorical_features_info']
        
        # Initialize model with updated features count
        model = SAINT(
            num_numerical_features=11,  # Updated based on new features
            categorical_features_info=categorical_features_info,
            d_model=128,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            num_classes=2
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, scaler, categorical_features_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_input(data):
    """Preprocess input data for prediction with new features"""
    # Prepare numerical features based on your new requirements
    numerical_features = np.array([
        data['migrant_tenure'] if data['migrant_status'] else 0,
        data['household_size'],
        data['disabled_count'] if data['has_disability'] else 0,
        data['chronic_ill_count'] if data['has_chronic_illness'] else 0,
        data['asset_count'] if data['has_assets'] else 0,
        data['asset_value'] if data['has_assets'] else 0,
        data['monthly_savings'] if data['has_savings'] else 0,
        data['monthly_income'],
        data['income_per_head'],
        data['loan_count'] if data['has_past_loans'] else 0,
        data['outstanding_amount'] if data['has_running_loans'] else 0
    ]).reshape(1, -1)
    
    # Prepare categorical features
    categorical_features = np.array([
        1 if data['marital_status'] else 0,
        1 if data['migrant_status'] else 0,
        1 if data['has_disability'] else 0,
        1 if data['has_chronic_illness'] else 0,
        1 if data['has_assets'] else 0,
        1 if data['has_savings'] else 0,
        1 if data['has_past_loans'] else 0,
        1 if data['has_running_loans'] else 0,
        1 if data['has_under_5'] else 0,
        1 if data['has_over_50'] else 0,
        1 if data['has_18_to_50'] else 0
    ]).reshape(1, -1)
    
    return numerical_features, categorical_features

def predict_aid_approval(model, scaler, numerical_features, categorical_features):
    """Make prediction using the trained model"""
    # Normalize numerical features
    numerical_features_scaled = scaler.transform(numerical_features)
    
    # Convert to tensors
    numerical_tensor = torch.FloatTensor(numerical_features_scaled)
    categorical_tensor = torch.LongTensor(categorical_features)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(numerical_tensor, categorical_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1)
    
    return predicted_class.item(), probabilities[0].numpy()

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Ultra-Poor Aid Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, categorical_features_info = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure 'saint_aid_prediction_model.pth' is in the same directory.")
        st.info("Make sure to save your trained model using the provided training script.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for information
    with st.sidebar:
        st.markdown("### ℹ️ About This System")
        st.markdown("""
        This AI-powered system uses a SAINT (Self-Attention and Intersample Attention Transformer) 
        neural network to predict aid approval for ultra-poor applicants.
        
        **Model Performance:**
        - Accuracy: ~95%
        - Architecture: Transformer-based
        - Features: Mixed numerical & categorical data
        """)
        
        st.markdown("### 📊 Prediction Factors")
        st.markdown("""
        The model considers:
        - Personal demographics
        - Household composition
        - Health conditions
        - Financial status
        - Asset ownership
        - Loan history
        """)
    
    # Main form
    st.markdown('<h2 class="sub-header">📝 Applicant Information</h2>', unsafe_allow_html=True)
    
    # Personal Information Section
    st.markdown("#### 👤 Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", placeholder="Enter full name")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        
    with col2:
        migrant_status = st.selectbox("Are you a migrant?", ['No','Yes'])
        migrant_tenure = 0
        if migrant_status == 'Yes':
            migrant_tenure = st.number_input("Migrant Tenure (years)", min_value=0, value=0)
        household_size = st.number_input("Number of people in household", min_value=1, value=4)
    
    # Health Information Section
    st.markdown("#### 🏥 Health Information")
    col1, col2 = st.columns(2)
    
    with col1:
        has_disability = st.selectbox("Is there anyone with disability in household?",['No','Yes'])
        disabled_count = 0
        if has_disability == 'Yes':
            disabled_count = st.number_input("Number of disabled people", min_value=1, value=1)
    
    with col2:
        has_chronic_illness = st.selectbox("Is there anyone chronically ill in household?",['No','Yes'])
        chronic_ill_count = 0
        if has_chronic_illness = 'Yes':
            chronic_ill_count = st.number_input("Number of chronically ill people", min_value=1, value=1)
    
    # Household Demographics Section
    st.markdown("#### 👨‍👩‍👧‍👦 Household Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        has_under_5 = st.selectbox("Are there people under 5 years old?",['No','Yes'])
    with col2:
        has_over_50 = st.selectbox("Are there people over 50 years old?",['No','Yes'])
    with col3:
        has_18_to_50 = st.selectbox("Are there people between 18-50 years old?",['No','Yes'])
    
    # Financial Information Section
    st.markdown("#### 💰 Financial Information")
    
    # Assets
    has_assets = st.selectbox("Does the family have assets?",['No','Yes'])
    asset_count = 0
    asset_value = 0
    if has_assets == 'Yes':
        col1, col2 = st.columns(2)
        with col1:
            asset_count = st.number_input("Number of assets", min_value=1, value=1)
        with col2:
            asset_value = st.number_input("Total value of assets (BDT)", min_value=0.0, value=0.0, format="%.2f")
    
    # Savings
    has_savings = st.selectbox("Does the family have savings?",['No','Yes'])
    monthly_savings = 0
    if has_savings == 'Yes':
        monthly_savings = st.number_input("Amount saved per month (BDT)", min_value=0.0, value=0.0, format="%.2f")
    
    # Income
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("Monthly Income (BDT)", min_value=0.0, value=0.0, format="%.2f")
    with col2:
        income_per_head = st.number_input("Monthly Income per Head (BDT)", min_value=0.0, value=0.0, format="%.2f")
    
    # Loan Information Section
    st.markdown("#### 📋 Loan Information")
    
    # Past loans
    has_past_loans = st.selectbox("Have you taken loans in the past?",['No','Yes'])
    loan_count = 0
    if has_past_loans == 'Yes':
        loan_count = st.number_input("Number of loans taken", min_value=1, value=1)
    
    # Running loans
    has_running_loans = st.selectbox("Do you have running loans?",['No','Yes'])
    outstanding_amount = 0
    if has_running_loans == 'Yes':
        outstanding_amount = st.number_input("Outstanding amount (BDT)", min_value=0.0, value=0.0, format="%.2f")
    
    # Prediction button
    st.markdown("---")
    
    if st.button("Predict Aid Approval", type="primary", use_container_width=True):
        if not name:
            st.error("Please enter the applicant's name.")
            return
        
        # Prepare data
        data = {
            'name': name,
            'age': age,
            'marital_status': marital_status,
            'migrant_status': migrant_status,
            'migrant_tenure': migrant_tenure,
            'household_size': household_size,
            'has_disability': has_disability,
            'disabled_count': disabled_count,
            'has_chronic_illness': has_chronic_illness,
            'chronic_ill_count': chronic_ill_count,
            'has_under_5': has_under_5,
            'has_over_50': has_over_50,
            'has_18_to_50': has_18_to_50,
            'has_assets': has_assets,
            'asset_count': asset_count,
            'asset_value': asset_value,
            'has_savings': has_savings,
            'monthly_savings': monthly_savings,
            'monthly_income': monthly_income,
            'income_per_head': income_per_head,
            'has_past_loans': has_past_loans,
            'loan_count': loan_count,
            'has_running_loans': has_running_loans,
            'outstanding_amount': outstanding_amount
        }
        
        # Preprocess and predict
        numerical_features, categorical_features = preprocess_input(data)
        prediction, probabilities = predict_aid_approval(model, scaler, numerical_features, categorical_features)
        
        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Display applicant summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👤 Applicant", name)
            st.metric("🎂 Age", f"{age} years")
        
        with col2:
            st.metric("👥 Household Size", f"{household_size} members")
            st.metric("💰 Monthly Income", f"BDT {monthly_income:,.2f}")
        
        with col3:
            st.metric("🏠 Has Assets", "Yes" if has_assets == 'Yes' else "No")
            st.metric("📋 Has Loans", "Yes" if has_past_loans == 'Yes' else "No")
        
        # Prediction result
        if prediction == 1:  # Aid approved
            st.markdown(f'''
            <div class="prediction-box approved">
                ✅ AID APPROVED
                <br>
                Confidence: {probabilities[1]:.1%}
            </div>
            ''', unsafe_allow_html=True)
        else:  # Aid not approved
            st.markdown(f'''
            <div class="prediction-box rejected">
                ❌ AID NOT APPROVED
                <br>
                Confidence: {probabilities[0]:.1%}
            </div>
            ''', unsafe_allow_html=True)
        
        # Detailed probabilities
        st.markdown("#### 📊 Detailed Prediction Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("🚫 Not Approved", f"{probabilities[0]:.1%}")
        
        with prob_col2:
            st.metric("✅ Approved", f"{probabilities[1]:.1%}")
        
        # Additional insights
        st.markdown("#### 💡 Key Factors Analysis")
        
        insights = []
        
        if income_per_head < 2000:
            insights.append("🔴 Very low income per household member")
        elif income_per_head < 5000:
            insights.append("🟡 Low income per household member")
        
        if has_assets and asset_value > 10000:
            insights.append("🟢 Significant asset ownership")
        elif not has_assets:
            insights.append("🔴 No asset ownership")
        
        if has_running_loans and outstanding_amount > monthly_income * 3:
            insights.append("🔴 High debt burden relative to income")
        elif not has_running_loans:
            insights.append("🟢 No current debt obligations")
        
        if household_size > 5:
            insights.append("🔴 Large household size")
        elif household_size <= 2:
            insights.append("🟢 Small household size")
        
        if has_disability or has_chronic_illness:
            insights.append("🔴 Health challenges in household")
        
        if has_under_5:
            insights.append("🟡 Has young children requiring care")
        
        if has_over_50:
            insights.append("🟡 Has elderly members requiring care")
        
        if migrant_status and migrant_tenure < 2:
            insights.append("🔴 Recent migrant status")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        
        # Export results
        if st.button("📁 Export Results", type="secondary"):
            result_data = {
                'Name': name,
                'Age': age,
                'Marital_Status': marital_status,
                'Migrant_Status': migrant_status,
                'Household_Size': household_size,
                'Monthly_Income': monthly_income,
                'Income_Per_Head': income_per_head,
                'Has_Assets': has_assets,
                'Asset_Value': asset_value,
                'Has_Disability': has_disability,
                'Has_Chronic_Illness': has_chronic_illness,
                'Has_Savings': has_savings,
                'Has_Past_Loans': has_past_loans,
                'Has_Running_Loans': has_running_loans,
                'Outstanding_Amount': outstanding_amount,
                'Prediction': 'Approved' if prediction == 1 else 'Not Approved',
                'Approval_Probability': f"{probabilities[1]:.1%}",
                'Rejection_Probability': f"{probabilities[0]:.1%}",
                'Prediction_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result_df = pd.DataFrame([result_data])
            csv = result_df.to_csv(index=False)
            
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"aid_prediction_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":

    main()
