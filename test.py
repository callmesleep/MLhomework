# 机器学习结课项目：肥胖风险预测
# 姓名：沙修竹
# 学号：20225843

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap
import warnings
warnings.filterwarnings('ignore')

class ObesityRiskPrediction:
    def __init__(self, data_path=None):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.feature_names = []
        self.data_path = data_path
    
    def load_data(self):
        """加载数据集"""
        if self.data_path:
            try:
                self.df = pd.read_csv(self.data_path)
            except:
                print("Failed to load local data, using built-in sample logic")
        else :
            print("Failed to load local data, using built-in sample logic")
        print("Data basic information:")
        self.df.info()
        print("\nData statistical description:")
        print(self.df.describe())
        print("\nTarget variable distribution:")
        print(self.df['0be1dad'].value_counts(normalize=True))
        return self.df
    
    def preprocess_data(self):
        """Data preprocessing: handle missing values, encode categorical variables, feature engineering"""
        # 检查必要的列是否存在
        required_columns = ['Gender', 'Age', 'Height', 'Weight', '0be1dad', 
                        'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 
                        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"Warning: Missing some required columns: {missing_columns}")
            print("Proceeding with available columns...")

        for col in required_columns:
            if col not in self.df.columns:
                if col == 'family_history_with_overweight':
                    print(f"Creating placeholder column for {col}")
                    self.df[col] = np.random.choice(['yes', 'no'], len(self.df))
                elif col == 'SMOKE':
                    print(f"Creating placeholder column for {col}")
                    self.df[col] = np.random.choice(['yes', 'no'], len(self.df))
                elif col == 'FAF':
                    print(f"Creating placeholder column for {col}")
                    self.df[col] = np.random.uniform(0, 3, len(self.df))
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # 检查缺失值
        missing_values = self.df.isnull().sum()
        print("Missing value statistics:")
        print(missing_values[missing_values > 0])

        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        num_imputer = SimpleImputer(strategy='mean')
        self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
        
        # 特征工程：计算BMI
        self.df['BMI'] = self.df['Weight'] / ((self.df['Height']/100) ** 2)

        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                            'SMOKE', 'SCC', 'CALC', 'MTRANS']
        categorical_cols = [col for col in categorical_cols if col in self.df.columns]
        print("\nColumn types before encoding:")
        print(self.df[categorical_cols].dtypes)
        le = LabelEncoder()
        for col in categorical_cols:
            print(f"Encoding column: {col}")
            print(f"Unique values before encoding: {self.df[col].unique()}")
            self.df[col] = le.fit_transform(self.df[col])
            print(f"Unique values after encoding: {self.df[col].unique()}")

        obesity_order = ['Underweight', 'Normal_Weight', 'Overweight_Level_I', 
                     'Overweight_Level_II', 'Obese_Level_I', 'Obese_Level_II', 'Obese_Level_III']
    
        actual_classes = self.df['0be1dad'].unique()
        if set(actual_classes) == set(obesity_order):
            print("Using ordered encoding for obesity levels")
            self.df['Obesity_label'] = pd.Categorical(self.df['0be1dad'], categories=obesity_order, ordered=True)
            self.obesity_mapping = {cat: code for code, cat in enumerate(obesity_order)}  
            self.df['Obesity_label'] = self.df['0be1dad'].cat.codes
        else:
            print("Actual classes do not match预设顺序, using standard label encoding")
            le = LabelEncoder()
            self.df['Obesity_label'] = le.fit_transform(self.df['0be1dad'])
            self.obesity_mapping = {cat: code for code, cat in enumerate(le.classes_)}  
        actual_classes = sorted(self.df['0be1dad'].unique())
        print(f"Actual obesity classes: {actual_classes}")
        
        if set(actual_classes) == set(obesity_order):
            print("Using ordered encoding for obesity levels")
            self.df['0be1dad'] = pd.Categorical(self.df['0be1dad'], categories=obesity_order, ordered=True)
            self.df['0be1dad'] = self.df['0be1dad'].cat.codes
        else:
            print("Actual classes do not match预设顺序, using standard label encoding")
            print(f"Actual classes: {actual_classes}")
            self.df['0be1dad'] = le.fit_transform(self.df['0be1dad'])
        
        # 选择特征
        feature_cols = ['Gender', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CAEC', 
                        'CH2O', 'SCC', 'FAVC', 'TUE', 'CALC', 'MTRANS', 'BMI',
                        'family_history_with_overweight', 'SMOKE', 'FAF']
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        self.feature_names = feature_cols
        self.X = self.df[feature_cols]
        self.y = self.df['Obesity_label']
        print("\nFeature data types before scaling:")
        print(self.X.dtypes)

        non_numeric_cols = self.X.select_dtypes(exclude=['float64', 'int64']).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Non-numeric columns found before scaling: {non_numeric_cols}")
            print("Data types:")
            print(self.X[non_numeric_cols].dtypes)
            print("Unique values:")
            for col in non_numeric_cols:
                print(f"{col}: {self.X[col].unique()}")
        
        # 特征标准化
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        print(f"Training set shape: {self.X_train.shape}, Test set shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def exploratory_data_analysis(self):
        """Exploratory data analysis: visualize the relationship between features and obesity risk"""
        # BMI与肥胖等级的关系
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='0be1dad', y='BMI', data=self.df, palette='plasma')
        plt.title('Relationship between BMI and Obesity Level')
        plt.xlabel('Obesity Level')
        plt.ylabel('BMI Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bmi_vs_obesity.png')
        plt.close()
        
        # 性别与肥胖等级的关系
        plt.figure(figsize=(10, 6))
        gender_map = {0: 'Female', 1: 'Male'}
        self.df['Gender_label'] = self.df['Gender'].map(gender_map)
        if hasattr(self, 'obesity_mapping'):
            code_to_obesity = {v: k for k, v in self.obesity_mapping.items()}  
            self.df['Obesity_label'] = self.df['0be1dad'].map(code_to_obesity)
        else:
            self.df['Obesity_label'] = self.df['0be1dad'].astype(str)
        sns.countplot(x='Gender_label', hue='Obesity_label', data=self.df, palette='Set1')
        plt.title('Obesity Level by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig('gender_vs_obesity.png')
        plt.close()
        
        # 特征相关性热力图
        corr = self.df[self.feature_names + ['0be1dad']].corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    def build_models(self):
        """构建多种机器学习模型"""
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42),
            "Neural Network": self._build_neural_network()
        }
        return self.models
    
    def _build_neural_network(self):
        """神经网络"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(7, activation='softmax')  # 7 obesity levels
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    
    def train_models(self):
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            if name == "Neural Network":
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                y_pred_proba = model.predict(self.X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            # 计算准确率和分类报告
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            cm = confusion_matrix(self.y_test, y_pred)
            
            self.model_results[name] = {
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": cm,
                "model": model,
                "y_pred": y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(pd.DataFrame(report).T)
        return self.model_results
    
    def evaluate_models(self):
        # 模型准确率比较
        accuracies = {name: result["accuracy"] for name, result in self.model_results.items()}
        results_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
        
        # 准确率可视化
        plt.figure(figsize=(12, 6))
        results_df.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'plum', 'orange', 'teal'])
        plt.title('Accuracy Comparison of Different Models')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png')
        plt.close()
        
        # 最佳模型的混淆矩阵和ROC曲线
        best_model_name = max(self.model_results, key=lambda x: self.model_results[x]["accuracy"])
        best_model = self.model_results[best_model_name]["model"]
        y_pred = self.model_results[best_model_name]["y_pred"]
        cm = self.model_results[best_model_name]["confusion_matrix"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix of {best_model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        y_test_binary = (self.y_test >= 3).astype(int)  # 肥胖等级大于等于3被视为正类
        y_pred_binary = (y_pred >= 3).astype(int)
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.close()
        
        return results_df
    
    def explain_model(self):
        """SHAP值分析模型：随机森林"""
        rf_model = self.model_results["Random Forest"]["model"]
        print("Calculating SHAP values, this may take some time...")
        explainer = shap.Explainer(rf_model)
        shap_values = explainer(self.X_test[:100])
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_test[:100], feature_names=self.feature_names, plot_type="bar")
        plt.title('SHAP Value Analysis of Feature Importance')
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        
        plt.figure(figsize=(14, 8))
        explanation = shap.Explanation(
            values=shap_values.values[0][0],  
            base_values=shap_values.base_values[0][0], 
            data=self.X_test[0],
            feature_names=self.feature_names
        )
        shap.plots.waterfall(explanation)
        plt.title('SHAP Explanation for a Single Sample')
        plt.tight_layout()
        plt.savefig('shap_waterfall.png')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            "BMI",  # Feature name
            shap_values.values[:, :, 0], 
            self.X_test[:100],
            feature_names=self.feature_names
        )
        plt.title('SHAP Dependence Plot for BMI Feature')
        plt.tight_layout()
        plt.savefig('shap_dependence.png')
        plt.close()
        
        return shap_values
    
    def summarize_results(self):
        # 找最优模型
        best_model_name = max(self.model_results, key=lambda x: self.model_results[x]["accuracy"])
        best_accuracy = self.model_results[best_model_name]["accuracy"]
        
        print("\n="*50)
        print("Project Summary")
        print("="*50)
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    project = ObesityRiskPrediction("./dataset.csv")  # Pass data path if using real data
    print("="*50)
    print("Step 1: Load Obesity Risk Dataset")
    project.load_data()
    print("="*50)
    print("Step 2: Data Preprocessing")
    project.preprocess_data()
    print("="*50)
    print("Step 3: Exploratory Data Analysis")
    project.exploratory_data_analysis()
    print("="*50)
    print("Step 4: Build Machine Learning Models")
    project.build_models()
    print("="*50)
    print("Step 5: Train and Evaluate Models")
    project.train_models()
    print("="*50)
    print("Step 6: Model Performance Visualization")
    project.evaluate_models()
    print("="*50)
    print("Step 7: Model Explanation (SHAP Value Analysis)")
    project.explain_model()  
    print("="*50)
    project.summarize_results()
    print("="*50)
    print("Obesity Risk Prediction Project completed! All visualization results saved as image files.")