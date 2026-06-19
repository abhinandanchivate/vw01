# ============================================================
# NAIVE BAYES AUTOMOBILE COMPLAINT CLASSIFICATION
# Complete end-to-end case study solution
# ============================================================

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
    recall_score,
    top_k_accuracy_score
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)
from sklearn.naive_bayes import (
    ComplementNB,
    MultinomialNB
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

ROOT = Path.cwd()

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"

for folder in [
    DATA_DIR,
    ARTIFACT_DIR,
    OUTPUT_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


DATASET_PATH = (
    DATA_DIR /
    "automobile_complaints.csv"
)

MODEL_PATH = (
    ARTIFACT_DIR /
    "complaint_classifier_pipeline.joblib"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR /
    "classification_report.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR /
    "model_comparison.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

MISSED_SAFETY_PATH = (
    OUTPUT_DIR /
    "missed_safety_complaints.csv"
)

PREDICTION_PATH = (
    OUTPUT_DIR /
    "new_complaint_predictions.csv"
)

METRICS_PATH = (
    OUTPUT_DIR /
    "model_metrics.csv"
)


# ============================================================
# 2. SAMPLE DATASET CONFIGURATION
# ============================================================

CATEGORY_CONFIG = {

    "General Service": {
        "count": 1200,

        "subjects": [
            "Regular service request",
            "Service appointment",
            "Maintenance query",
            "Periodic service required"
        ],

        "complaints": [
            "I want to book routine service for my vehicle",
            "Please schedule regular maintenance and oil change",
            "The vehicle is due for periodic service",
            "I need a general inspection before a long trip",
            "Please arrange normal maintenance for my car",
            "I want to check the next available service appointment"
        ]
    },

    "Engine and Mechanical": {
        "count": 650,

        "subjects": [
            "Engine noise",
            "Mechanical problem",
            "Overheating issue",
            "Gear and clutch problem"
        ],

        "complaints": [
            "The engine makes a loud knocking sound during acceleration",
            "The engine is overheating and losing power",
            "There is heavy vibration from the engine compartment",
            "Gear shifting is difficult and the clutch is slipping",
            "Oil is leaking from below the engine",
            "The engine warning appears and the vehicle loses power"
        ]
    },

    "Electrical and Battery": {
        "count": 500,

        "subjects": [
            "Battery issue",
            "Electrical warning",
            "Vehicle not starting",
            "Charging problem"
        ],

        "complaints": [
            "The battery drains overnight and the vehicle does not start",
            "The charging warning appears and headlights are flickering",
            "The starter motor is not responding",
            "The dashboard electrical warning appears repeatedly",
            "The battery is not charging properly",
            "The central locking and electrical controls are not working"
        ]
    },

    "Billing and Payment": {
        "count": 400,

        "subjects": [
            "Incorrect invoice",
            "Refund pending",
            "Duplicate charge",
            "Payment issue"
        ],

        "complaints": [
            "I was charged twice for the same repair",
            "The invoice contains parts that were not replaced",
            "Payment was deducted but the receipt was not generated",
            "The refund for cancelled service is still pending",
            "The final bill is higher than the approved estimate",
            "An incorrect labour charge has been added to the invoice"
        ]
    },

    "Warranty Claim": {
        "count": 300,

        "subjects": [
            "Warranty request",
            "Claim rejected",
            "Warranty replacement",
            "Coverage problem"
        ],

        "complaints": [
            "My warranty replacement request was rejected",
            "The failed component should be covered under warranty",
            "Please reopen the warranty claim for battery replacement",
            "The service centre denied warranty coverage",
            "My vehicle is under warranty but I was asked to pay",
            "The warranty claim has been pending for several weeks"
        ]
    },

    "Delivery Delay": {
        "count": 250,

        "subjects": [
            "Vehicle delivery delayed",
            "Repair pending",
            "Promised date missed",
            "Vehicle not delivered"
        ],

        "complaints": [
            "Vehicle delivery has been pending for many days",
            "The workshop missed the promised delivery date",
            "My vehicle is still not ready after repeated follow up",
            "Repair is complete but delivery is continuously delayed",
            "The service centre is not confirming the delivery date",
            "My vehicle has remained at the workshop for two weeks"
        ]
    },

    "Safety Critical": {
        "count": 200,

        "subjects": [
            "Urgent safety issue",
            "Brake failure",
            "Fire risk",
            "Steering failure"
        ],

        "complaints": [
            "The brake pedal failed while driving on the highway",
            "The steering locked suddenly and I nearly had an accident",
            "Smoke and fire came from the engine compartment",
            "The vehicle accelerated without pressing the pedal",
            "The brake did not respond and the vehicle crossed the signal",
            "Fuel is leaking and there is a strong risk of fire"
        ]
    }
}


GENERIC_CONTEXT = [
    "Please resolve this quickly",
    "The problem happened again today",
    "I have already contacted the service centre",
    "This problem is affecting my daily travel",
    "The problem started after the last service",
    "I have followed up several times",
    "Please contact me as soon as possible"
]


SPELLING_REPLACEMENTS = {
    "battery": "batery",
    "engine": "engne",
    "brake": "break",
    "warranty": "waranty",
    "delivery": "delivary",
    "vehicle": "vehical"
}


# ============================================================
# 3. GENERATE SAMPLE DATASET
# ============================================================

def introduce_spelling_error(
    text,
    random_generator
):
    """
    Occasionally introduces a spelling error to make
    the synthetic dataset more realistic.
    """

    if random_generator.random() > 0.07:
        return text

    for correct_word, incorrect_word in (
        SPELLING_REPLACEMENTS.items()
    ):
        if correct_word in text.lower():
            return re.sub(
                correct_word,
                incorrect_word,
                text,
                flags=re.IGNORECASE
            )

    return text


def generate_dataset(
    random_state=RANDOM_STATE
):
    """
    Generates an imbalanced automobile complaint dataset.
    """

    random_generator = np.random.default_rng(
        random_state
    )

    rows = []

    complaint_number = 100000

    for category, config in (
        CATEGORY_CONFIG.items()
    ):

        for _ in range(config["count"]):

            complaint_number += 1

            subject = random_generator.choice(
                config["subjects"]
            )

            complaint = random_generator.choice(
                config["complaints"]
            )

            # Add normal customer language
            if random_generator.random() < 0.70:
                complaint = (
                    complaint
                    + ". "
                    + random_generator.choice(
                        GENERIC_CONTEXT
                    )
                )

            # Add an overlapping topic
            if (
                random_generator.random() < 0.08
                and category != "General Service"
            ):
                complaint = (
                    complaint
                    + ". I also want to book "
                    + "a regular service"
                )

            complaint = introduce_spelling_error(
                complaint,
                random_generator
            )

            # Generate category-related numeric patterns
            if category == "Safety Critical":

                vehicle_age = int(
                    np.clip(
                        random_generator.normal(
                            8,
                            3
                        ),
                        1,
                        18
                    )
                )

                mileage = int(
                    np.clip(
                        random_generator.normal(
                            110000,
                            35000
                        ),
                        5000,
                        250000
                    )
                )

                previous_complaints = int(
                    np.clip(
                        random_generator.poisson(3),
                        0,
                        15
                    )
                )

            elif category in [
                "Engine and Mechanical",
                "Electrical and Battery"
            ]:

                vehicle_age = int(
                    np.clip(
                        random_generator.normal(
                            6,
                            3
                        ),
                        1,
                        18
                    )
                )

                mileage = int(
                    np.clip(
                        random_generator.normal(
                            85000,
                            30000
                        ),
                        5000,
                        250000
                    )
                )

                previous_complaints = int(
                    np.clip(
                        random_generator.poisson(2),
                        0,
                        15
                    )
                )

            else:

                vehicle_age = int(
                    np.clip(
                        random_generator.normal(
                            4,
                            2.5
                        ),
                        1,
                        18
                    )
                )

                mileage = int(
                    np.clip(
                        random_generator.normal(
                            55000,
                            25000
                        ),
                        5000,
                        250000
                    )
                )

                previous_complaints = int(
                    np.clip(
                        random_generator.poisson(1),
                        0,
                        15
                    )
                )

            if (
                category == "Warranty Claim"
                and random_generator.random() < 0.85
            ):
                warranty_status = "Yes"
            else:
                warranty_status = (
                    random_generator.choice(
                        ["Yes", "No"],
                        p=[0.45, 0.55]
                    )
                )

            rows.append({
                "complaint_id": (
                    f"CMP{complaint_number}"
                ),

                "customer_id": (
                    f"CUST"
                    f"{random_generator.integers(1000, 9999)}"
                ),

                "vehicle_model": (
                    random_generator.choice([
                        "Hatchback",
                        "Sedan",
                        "SUV",
                        "Commercial"
                    ])
                ),

                "vehicle_age_years": vehicle_age,

                "mileage_km": mileage,

                "subject_line": subject,

                "complaint_text": complaint,

                "communication_channel": (
                    random_generator.choice([
                        "Email",
                        "Mobile App",
                        "Website",
                        "Call Centre"
                    ])
                ),

                "previous_complaints": (
                    previous_complaints
                ),

                "days_since_last_service": int(
                    np.clip(
                        random_generator.normal(
                            180,
                            110
                        ),
                        1,
                        900
                    )
                ),

                "vehicle_under_warranty": (
                    warranty_status
                ),

                "customer_region": (
                    random_generator.choice([
                        "West",
                        "North",
                        "South",
                        "East"
                    ])
                ),

                "complaint_category": category
            })

    dataset = pd.DataFrame(rows)

    dataset = dataset.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    # Introduce missing values
    missing_generator = np.random.default_rng(
        random_state + 1
    )

    columns_with_missing_values = [
        "mileage_km",
        "days_since_last_service",
        "vehicle_under_warranty",
        "customer_region"
    ]

    for column in columns_with_missing_values:

        missing_count = int(
            len(dataset) * 0.02
        )

        missing_indexes = (
            missing_generator.choice(
                dataset.index,
                size=missing_count,
                replace=False
            )
        )

        dataset.loc[
            missing_indexes,
            column
        ] = np.nan

    # Create a small number of blank complaints
    blank_count = int(
        len(dataset) * 0.005
    )

    blank_indexes = missing_generator.choice(
        dataset.index,
        size=blank_count,
        replace=False
    )

    dataset.loc[
        blank_indexes,
        "complaint_text"
    ] = ""

    return dataset


dataset = generate_dataset()

dataset.to_csv(
    DATASET_PATH,
    index=False
)

print("Dataset created successfully.")
print("Dataset path:", DATASET_PATH)
print("Dataset shape:", dataset.shape)

print("\nCategory distribution:")

print(
    dataset[
        "complaint_category"
    ].value_counts()
)


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def prepare_dataframe(dataframe):
    """
    Creates the combined complaint text and performs
    basic text cleaning.

    Negation words such as 'not' are preserved.
    """

    prepared = dataframe.copy()

    subject = prepared.get(
        "subject_line",
        pd.Series(
            "",
            index=prepared.index
        )
    ).fillna("")

    complaint = prepared.get(
        "complaint_text",
        pd.Series(
            "",
            index=prepared.index
        )
    ).fillna("")

    combined_text = (
        subject.astype(str)
        + " "
        + complaint.astype(str)
    )

    combined_text = (
        combined_text
        .str.lower()
        .str.replace(
            r"https?://\S+|www\.\S+",
            " ",
            regex=True
        )
        .str.replace(
            r"[^a-z0-9\s]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    prepared[
        "combined_text"
    ] = combined_text

    return prepared


dataset = prepare_dataframe(
    dataset
)


# Remove records containing no usable text
dataset = dataset[
    dataset["combined_text"].str.len() > 0
].copy()


# Remove duplicate complaint-category combinations
dataset = dataset.drop_duplicates(
    subset=[
        "combined_text",
        "complaint_category"
    ]
).reset_index(drop=True)


print(
    "\nDataset after cleaning:",
    dataset.shape
)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

TARGET_COLUMN = "complaint_category"

TEXT_COLUMN = "combined_text"


CATEGORICAL_FEATURES = [
    "communication_channel",
    "vehicle_under_warranty",
    "vehicle_model",
    "customer_region"
]


NUMERICAL_FEATURES = [
    "vehicle_age_years",
    "mileage_km",
    "previous_complaints",
    "days_since_last_service"
]


FEATURE_COLUMNS = [
    "complaint_id",
    "subject_line",
    "complaint_text",
    TEXT_COLUMN
] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES


X = dataset[
    FEATURE_COLUMNS
].copy()

y = dataset[
    TARGET_COLUMN
].copy()


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


print("\nTraining distribution:")

print(
    y_train.value_counts()
)


# ============================================================
# 7. TEXT TRANSFORMER
# ============================================================

text_transformer = TfidfVectorizer(

    # Preserve words such as not, never and no
    stop_words=None,

    lowercase=True,

    max_features=20000,

    sublinear_tf=True
)


# ============================================================
# 8. CATEGORICAL PIPELINE
# ============================================================

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 9. NUMERICAL PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            MinMaxScaler()
        )
    ]
)


# Why MinMaxScaler?
#
# MultinomialNB and ComplementNB expect non-negative
# input values.
#
# StandardScaler can create negative values.
# MinMaxScaler converts values to the 0-to-1 range.


# ============================================================
# 10. COMBINE ALL PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "text",
            text_transformer,
            TEXT_COLUMN
        ),

        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES
        ),

        (
            "numerical",
            numeric_transformer,
            NUMERICAL_FEATURES
        )
    ],

    remainder="drop"
)


# ============================================================
# 11. CREATE MODEL PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[

        (
            "preprocess",
            preprocessor
        ),

        (
            "model",
            ComplementNB()
        )
    ]
)


# ============================================================
# 12. SAFETY-CLASS RECALL SCORER
# ============================================================

safety_recall_scorer = make_scorer(
    recall_score,

    labels=[
        "Safety Critical"
    ],

    average="macro",

    zero_division=0
)


# ============================================================
# 13. HYPERPARAMETER GRID
# ============================================================

parameter_grid = [

    {
        "model": [
            MultinomialNB()
        ],

        "model__alpha": [
            0.1,
            0.5,
            1.0
        ],

        "preprocess__text__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "preprocess__text__min_df": [
            2
        ]
    },

    {
        "model": [
            ComplementNB()
        ],

        "model__alpha": [
            0.1,
            0.5,
            1.0
        ],

        "preprocess__text__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "preprocess__text__min_df": [
            2
        ]
    }
]


# ============================================================
# 14. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=model_pipeline,

    param_grid=parameter_grid,

    scoring={
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
        "safety_recall": (
            safety_recall_scorer
        )
    },

    # Select final model using macro F1
    refit="macro_f1",

    cv=3,

    n_jobs=-1,

    verbose=1,

    return_train_score=False
)


print(
    "\nTraining and tuning "
    "Naive Bayes models..."
)


grid_search.fit(
    X_train,
    y_train
)


best_model = (
    grid_search.best_estimator_
)


print("\nBest parameters:")

print(
    grid_search.best_params_
)


print(
    "\nBest cross-validation "
    "Macro F1:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 15. SAVE MODEL-COMPARISON RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)


comparison_columns = [
    "param_model",
    "param_model__alpha",
    "param_preprocess__text__ngram_range",
    "mean_test_accuracy",
    "mean_test_macro_f1",
    "mean_test_safety_recall",
    "rank_test_macro_f1"
]


model_comparison = cv_results[
    comparison_columns
].copy()


model_comparison[
    "param_model"
] = (
    model_comparison[
        "param_model"
    ].astype(str)
)


model_comparison = (
    model_comparison.sort_values(
        "rank_test_macro_f1"
    )
)


model_comparison.to_csv(
    MODEL_COMPARISON_PATH,
    index=False
)


print(
    "\nModel comparison:"
)

print(
    model_comparison.head(10)
    .to_string(index=False)
)


# ============================================================
# 16. TEST-DATA PREDICTIONS
# ============================================================

test_predictions = (
    best_model.predict(
        X_test
    )
)


test_probabilities = (
    best_model.predict_proba(
        X_test
    )
)


class_order = (
    best_model
    .named_steps["model"]
    .classes_
)


print(
    "\nClass order:",
    class_order
)


# ============================================================
# 17. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)


macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)


weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)


safety_recall = recall_score(
    y_test,
    test_predictions,

    labels=[
        "Safety Critical"
    ],

    average="macro",

    zero_division=0
)


top_2_accuracy = top_k_accuracy_score(
    y_test,
    test_probabilities,

    k=2,

    labels=class_order
)


print("\nTest metrics")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro F1:",
    round(macro_f1, 4)
)

print(
    "Weighted F1:",
    round(weighted_f1, 4)
)

print(
    "Safety Critical Recall:",
    round(safety_recall, 4)
)

print(
    "Top-2 Accuracy:",
    round(top_2_accuracy, 4)
)


metrics_dataframe = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Safety Critical Recall",
        "Top-2 Accuracy"
    ],

    "value": [
        accuracy,
        macro_f1,
        weighted_f1,
        safety_recall,
        top_2_accuracy
    ]
})


metrics_dataframe.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report_dictionary = (
    classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )
)


report_dataframe = pd.DataFrame(
    report_dictionary
).transpose()


report_dataframe.to_csv(
    CLASSIFICATION_REPORT_PATH
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y_test,
    test_predictions,
    labels=class_order
)


display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=class_order
)


fig, axis = plt.subplots(
    figsize=(12, 9)
)


display.plot(
    ax=axis,
    xticks_rotation=45,
    values_format="d"
)


axis.set_title(
    "Naive Bayes Automobile Complaint Classification"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. SAFETY-CRITICAL ERROR ANALYSIS
# ============================================================

safety_index = list(
    class_order
).index(
    "Safety Critical"
)


test_results = X_test.copy()


test_results[
    "actual_category"
] = y_test.values


test_results[
    "predicted_category"
] = test_predictions


test_results[
    "safety_probability"
] = (
    test_probabilities[
        :,
        safety_index
    ].round(4)
)


test_results[
    "prediction_correct"
] = (
    test_results[
        "actual_category"
    ]
    ==
    test_results[
        "predicted_category"
    ]
)


missed_safety_complaints = (
    test_results[
        (
            test_results[
                "actual_category"
            ]
            ==
            "Safety Critical"
        )
        &
        (
            test_results[
                "predicted_category"
            ]
            !=
            "Safety Critical"
        )
    ]
    .sort_values(
        "safety_probability",
        ascending=False
    )
)


missed_safety_complaints.to_csv(
    MISSED_SAFETY_PATH,
    index=False
)


print(
    "\nNumber of missed "
    "Safety Critical complaints:",
    len(missed_safety_complaints)
)


if len(
    missed_safety_complaints
) > 0:

    print(
        "\nMissed safety examples:"
    )

    print(
        missed_safety_complaints[
            [
                "complaint_id",
                "complaint_text",
                "actual_category",
                "predicted_category",
                "safety_probability"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 21. SAVE THE TRAINED PIPELINE
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


print(
    "\nSaved trained model to:",
    MODEL_PATH
)


# ============================================================
# 22. DEPARTMENT AND ACTION RULES
# ============================================================

DEPARTMENT_MAPPING = {

    "General Service":
        "Service Operations",

    "Engine and Mechanical":
        "Mechanical Diagnostics",

    "Electrical and Battery":
        "Electrical Diagnostics",

    "Billing and Payment":
        "Finance Department",

    "Warranty Claim":
        "Warranty Department",

    "Delivery Delay":
        "Workshop Operations",

    "Safety Critical":
        "Emergency Escalation Team"
}


DEFAULT_ACTIONS = {

    "General Service":
        "Schedule a normal service appointment",

    "Engine and Mechanical":
        "Assign to mechanical diagnostic team",

    "Electrical and Battery":
        "Assign to electrical diagnostic team",

    "Billing and Payment":
        "Send to billing investigation team",

    "Warranty Claim":
        "Validate warranty and claim eligibility",

    "Delivery Delay":
        "Escalate to workshop operations manager",

    "Safety Critical":
        "Contact customer immediately and arrange emergency assistance"
}


# ============================================================
# 23. NEW COMPLAINT PREDICTION FUNCTION
# ============================================================

def predict_complaints(
    trained_model,
    new_complaints
):
    """
    Predicts complaint classes, probabilities,
    departments, escalation level and manual-review status.
    """

    prepared_data = prepare_dataframe(
        new_complaints
    )

    model_predictions = (
        trained_model.predict(
            prepared_data
        )
    )

    probabilities = (
        trained_model.predict_proba(
            prepared_data
        )
    )

    model_classes = (
        trained_model
        .named_steps["model"]
        .classes_
    )

    safety_class_index = list(
        model_classes
    ).index(
        "Safety Critical"
    )

    result = prepared_data.copy()

    result[
        "model_prediction"
    ] = model_predictions

    result[
        "highest_probability"
    ] = (
        probabilities.max(
            axis=1
        ).round(4)
    )

    result[
        "safety_probability"
    ] = (
        probabilities[
            :,
            safety_class_index
        ].round(4)
    )

    # Save every class probability
    for index, category in enumerate(
        model_classes
    ):

        safe_column_name = (
            category.lower()
            .replace(" ", "_")
            .replace("&", "and")
        )

        result[
            f"probability_{safe_column_name}"
        ] = (
            probabilities[
                :,
                index
            ].round(4)
        )

    routed_categories = []
    departments = []
    manual_review_flags = []
    recommended_actions = []

    for row_index, row in (
        result.iterrows()
    ):

        model_category = row[
            "model_prediction"
        ]

        confidence = row[
            "highest_probability"
        ]

        safety_probability = row[
            "safety_probability"
        ]

        complaint_word_count = len(
            str(
                row["combined_text"]
            ).split()
        )

        routed_category = model_category

        manual_review = "No"

        # Rule 1: Strong safety probability
        if safety_probability >= 0.70:

            routed_category = (
                "Safety Critical"
            )

            department = (
                DEPARTMENT_MAPPING[
                    "Safety Critical"
                ]
            )

            action = (
                "Immediately contact the customer, "
                "advise the customer not to drive, "
                "arrange towing support and notify "
                "the regional safety manager"
            )

        # Rule 2: Possible safety complaint
        elif safety_probability >= 0.40:

            manual_review = "Yes"

            department = (
                "Safety Review Team"
            )

            action = (
                "Perform immediate manual safety "
                "review before normal department routing"
            )

        # Rule 3: Insufficient complaint information
        elif complaint_word_count < 5:

            manual_review = "Yes"

            department = (
                "Customer Support Review"
            )

            action = (
                "Contact the customer and request "
                "additional complaint information"
            )

        # Rule 4: Low model confidence
        elif confidence < 0.50:

            manual_review = "Yes"

            department = (
                "Manual Classification Team"
            )

            action = (
                "Manually classify the complaint "
                "because model confidence is low"
            )

        # Rule 5: Medium confidence
        elif confidence < 0.75:

            manual_review = "Yes"

            department = (
                DEPARTMENT_MAPPING[
                    model_category
                ]
            )

            action = (
                DEFAULT_ACTIONS[
                    model_category
                ]
                +
                " and verify the classification manually"
            )

        # Rule 6: High-confidence automatic routing
        else:

            department = (
                DEPARTMENT_MAPPING[
                    model_category
                ]
            )

            action = (
                DEFAULT_ACTIONS[
                    model_category
                ]
            )

        routed_categories.append(
            routed_category
        )

        departments.append(
            department
        )

        manual_review_flags.append(
            manual_review
        )

        recommended_actions.append(
            action
        )

    result[
        "routed_category"
    ] = routed_categories

    result[
        "predicted_department"
    ] = departments

    result[
        "manual_review_required"
    ] = manual_review_flags

    result[
        "recommended_action"
    ] = recommended_actions

    return result


# ============================================================
# 24. CREATE NEW COMPLAINTS
# ============================================================

new_complaints = pd.DataFrame({

    "complaint_id": [
        "NEW001",
        "NEW002",
        "NEW003",
        "NEW004",
        "NEW005",
        "NEW006"
    ],

    "subject_line": [
        "Steering locked",
        "Duplicate charge",
        "Battery warning",
        "Vehicle not delivered",
        "Regular maintenance",
        "Warranty denied"
    ],

    "complaint_text": [

        (
            "While driving at high speed the steering "
            "locked and I nearly had an accident"
        ),

        (
            "The service centre charged my card twice "
            "for the same repair"
        ),

        (
            "The battery warning appears and the "
            "vehicle is not starting"
        ),

        (
            "My vehicle has been in the workshop for "
            "twelve days and delivery is still pending"
        ),

        (
            "I need to book regular service and oil change"
        ),

        (
            "My car is under warranty but the replacement "
            "claim was rejected"
        )
    ],

    "communication_channel": [
        "Call Centre",
        "Email",
        "Mobile App",
        "Website",
        "Mobile App",
        "Email"
    ],

    "vehicle_under_warranty": [
        "Yes",
        "No",
        "Yes",
        "Yes",
        "No",
        "Yes"
    ],

    "vehicle_model": [
        "SUV",
        "Sedan",
        "Hatchback",
        "SUV",
        "Sedan",
        "Hatchback"
    ],

    "customer_region": [
        "West",
        "North",
        "South",
        "East",
        "West",
        "South"
    ],

    "vehicle_age_years": [
        7,
        3,
        5,
        4,
        2,
        3
    ],

    "mileage_km": [
        105000,
        42000,
        65000,
        50000,
        22000,
        35000
    ],

    "previous_complaints": [
        3,
        1,
        2,
        4,
        0,
        2
    ],

    "days_since_last_service": [
        310,
        80,
        180,
        150,
        290,
        120
    ]
})


# ============================================================
# 25. LOAD SAVED MODEL AND PREDICT
# ============================================================

loaded_model = joblib.load(
    MODEL_PATH
)


prediction_results = predict_complaints(
    loaded_model,
    new_complaints
)


prediction_results.to_csv(
    PREDICTION_PATH,
    index=False
)


display_columns = [
    "complaint_id",
    "model_prediction",
    "highest_probability",
    "safety_probability",
    "routed_category",
    "predicted_department",
    "manual_review_required",
    "recommended_action"
]


print(
    "\nNew complaint predictions:"
)

print(
    prediction_results[
        display_columns
    ].to_string(index=False)
)


print(
    "\nPredictions saved to:",
    PREDICTION_PATH
)


# ============================================================
# 26. FINAL FILE SUMMARY
# ============================================================

print("\nGenerated files:")

print(
    "1. Dataset:",
    DATASET_PATH
)

print(
    "2. Trained pipeline:",
    MODEL_PATH
)

print(
    "3. Classification report:",
    CLASSIFICATION_REPORT_PATH
)

print(
    "4. Model comparison:",
    MODEL_COMPARISON_PATH
)

print(
    "5. Model metrics:",
    METRICS_PATH
)

print(
    "6. Confusion matrix:",
    CONFUSION_MATRIX_PATH
)

print(
    "7. Missed safety complaints:",
    MISSED_SAFETY_PATH
)

print(
    "8. New complaint predictions:",
    PREDICTION_PATH
)
