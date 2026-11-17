from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Telco Customer Churn Predictor")

st.write(
    "Enter customer information below to predict "
    "whether the customer is likely to churn."
)

st.divider()

# Customer information form
st.header("Customer Information")


with st.form("customer_form"):

    col1, col2 = st.columns(2)

    # LEFT COLUMN — Customer and account details
    with col1:
        st.subheader("Customer Information")

        gender = st.selectbox("Gender", ["Male", "Female"])

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    # RIGHT COLUMN — Services and billing
    with col2:
        st.subheader("Services & Billing")

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


#pdf report
def create_pdf_report(report, prediction, churn_probability, risk_level):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    # Theme colors
    navy = colors.HexColor("#172D49")
    blue = colors.HexColor("#3769A5")
    light_blue = colors.HexColor("#EAF1FA")
    light_grey = colors.HexColor("#F3F5F8")
    dark_grey = colors.HexColor("#465366")
    green = colors.HexColor("#21865B")
    red = colors.HexColor("#C84646")
    orange = colors.HexColor("#D58A25")

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=27,
        textColor=colors.white,
        alignment=TA_LEFT
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=navy,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=dark_grey
    ))

    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=dark_grey
    ))

    content = []

    # 1. Header
    header = Table([
        [
            Paragraph(
                "CUSTOMER INSIGHTS<br/><br/>"
                "<font size='10'>TELCO CUSTOMER ANALYTICS</font>",
                styles["ReportTitle"]
            )
        ]
    ], colWidths=[174 * mm])

    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy),
        ("BOX", (0, 0), (-1, -1), 0, navy),
        ("LEFTPADDING", (0, 0), (-1, -1), 15),
        ("RIGHTPADDING", (0, 0), (-1, -1), 15),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))

    content.append(header)
    content.append(Spacer(1, 12))

    content.append(Paragraph(
        "Customer Churn Prediction Report",
        styles["Heading1"]
    ))

    content.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        styles["SmallText"]
    ))

    content.append(Spacer(1, 18))

    
    # 2. Prediction summary
    content.append(Paragraph(
        "Prediction Summary",
        styles["SectionTitle"]
    ))

    prediction_color = red if prediction == "Yes" else green

    label_style = ParagraphStyle(
        "SummaryLabel",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=dark_grey
    )

    value_style = ParagraphStyle(
        "SummaryValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=27,
        textColor=dark_grey
    )

    prediction_style = ParagraphStyle(
        "PredictionValue",
        parent=value_style,
        textColor=prediction_color
    )

    probability_card = Table([
        [
            Paragraph("PREDICTED CHURN", label_style),
            Paragraph("CHURN PROBABILITY", label_style),
            Paragraph("RISK LEVEL", label_style)
        ],
        [
            Paragraph(prediction.upper(), prediction_style),
            Paragraph(f"{churn_probability * 100:.2f}%", value_style),
            Paragraph(risk_level.upper(), value_style)
        ]
    ], colWidths=[58 * mm, 58 * mm, 58 * mm])

    probability_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_blue),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DFEC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))

    content.append(probability_card)
    content.append(Spacer(1, 12))


    
    # 3. Probability bar
    probability = max(0, min(churn_probability * 100, 100))

    bar_color = (
        green if risk_level == "Low"
        else orange if risk_level == "Medium"
        else red
    )

    total_width = 174 * mm

    filled_width = total_width * probability / 100
    filled_width = min(
        max(filled_width, 0.5 * mm),
        total_width - 0.5 * mm
    )

    remaining_width = total_width - filled_width

    bar = Table(
        [["", ""]],
        colWidths=[filled_width, remaining_width],
        rowHeights=[5 * mm]
    )

    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bar_color),
        ("BACKGROUND", (1, 0), (1, 0), light_grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    content.append(bar)
    content.append(Spacer(1, 20))


    # 4. Customer information
    content.append(Paragraph(
        "Customer Profile",
        styles["SectionTitle"]
    ))

    sections = {
        "Personal & Account Information": [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"
        ],
        "Services": [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ],
        "Billing & Contract": [
            "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
    }

    for section_name, fields in sections.items():

        content.append(Paragraph(
            section_name,
            styles["Heading3"]
        ))

        table_data = [["Attribute", "Customer Value"]]

        for field in fields:
            if field in report.columns:
                value = report.iloc[0][field]

                if field == "SeniorCitizen":
                    value = "Yes" if value == 1 else "No"

                if field in ["MonthlyCharges", "TotalCharges"]:
                    value = f"${float(value):,.2f}"

                if field == "tenure":
                    value = f"{value} months"

                table_data.append([field, str(value)])

        info_table = Table(
            table_data,
            colWidths=[70 * mm, 104 * mm]
        )

        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE1E8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, light_grey]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))

        content.append(info_table)
        content.append(Spacer(1, 12))

    # 5. Business recommendation
    content.append(Paragraph(
        "Recommended Next Step",
        styles["SectionTitle"]
    ))

    if prediction == "Yes":
        recommendation = (
            "This customer has been classified as likely to churn. "
            "Consider reviewing their service experience, contract "
            "options, and support requirements. A targeted retention "
            "offer may be worth evaluating."
        )
    else:
        recommendation = (
            "This customer has been classified as less likely to churn. "
            "Continue monitoring customer satisfaction and maintain "
            "consistent service quality."
        )

    recommendation_table = Table([
        [Paragraph(recommendation, styles["BodyTextCustom"])]
    ], colWidths=[174 * mm])

    recommendation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_blue),
        ("BOX", (0, 0), (-1, -1), 0.5, blue),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    content.append(recommendation_table)
    content.append(Spacer(1, 20))

    # 6. Disclaimer and footer
    content.append(HRFlowable(
        width="100%",
        thickness=0.5,
        color=colors.HexColor("#DCE1E8")
    ))

    content.append(Spacer(1, 8))

    content.append(Paragraph(
        "Disclaimer: This report contains predictions generated by a "
        "machine learning model. Predictions are estimates and should "
        "not be treated as guaranteed outcomes or the sole basis for "
        "business decisions.",
        styles["SmallText"]
    ))

    content.append(Spacer(1, 8))

    content.append(Paragraph(
        "Telco Customer Churn Prediction | Machine Learning Project",
        styles["SmallText"]
    ))

    pdf.build(content)

    buffer.seek(0)
    return buffer.getvalue()



if submitted:

    # Create a DataFrame using customer inputs
    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Predict churn probability
    churn_probability = model.predict_proba(customer_data)[0][1]

    # Apply our selected threshold
    threshold = 0.5

    if churn_probability >= threshold:
        prediction = "Yes"
    else:
        prediction = "No"

    # Determine customer risk level
    if churn_probability < 0.30:
        risk_level = "Low"
    elif churn_probability < 0.60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Prediction Result Dashboard

    st.divider()
    st.header("Prediction Analysis")

    with st.container(border=True):

        # Top section
        st.subheader("Customer Churn Assessment")

        result_col, probability_col = st.columns([1, 1])

        with result_col:
            st.caption("PREDICTED CHURN")

            if prediction == "Yes":
                st.error("## YES")
                st.caption("Customer is predicted to churn")
            else:
                st.success("## NO")
                st.caption("Customer is predicted to stay")

        with probability_col:
            st.caption("CHURN PROBABILITY")
            st.metric(
                label="Estimated probability",
                value=f"{churn_probability * 100:.2f}%"
            )

        st.divider()

        # Risk classification
        st.write("**Customer Risk Classification**")

        if risk_level == "Low":
            st.success("LOW RISK")
        elif risk_level == "Medium":
            st.warning("MEDIUM RISK")
        else:
            st.error("HIGH RISK")

        st.progress(float(churn_probability))

        st.caption(
            f"Churn probability: {churn_probability * 100:.2f}%"
        )

        st.divider()

        # Recommendation
        st.write("**Recommended Action**")

        if prediction == "Yes":
            st.write(
                "Consider reviewing this customer's service experience, "
                "contract options and support requirements. "
                "A targeted retention strategy may help."
            )
        else:
            st.write(
                "Continue monitoring customer satisfaction and "
                "maintain consistent service quality."
            )



    # Create prediction report
    report = customer_data.copy()

    report["Predicted_Churn"] = prediction
    report["Churn_Probability"] = round(
        churn_probability * 100, 2
    )
    report["Risk_Level"] = risk_level

    
    # Generate both report formats
    csv_data = report.to_csv(index=False)
    pdf_data = create_pdf_report(
        report,
        prediction,
        churn_probability,
        risk_level
    )

    # Download menu
    download_col, arrow_col = st.columns([5, 1])

    with download_col:
        st.download_button(
            label="⬇ Download Report",
            data=csv_data,
            file_name="churn_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
            on_click="ignore"
        )

    with arrow_col:
        with st.popover("▼", use_container_width=True):
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name="churn_prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
                on_click="ignore"
            )

            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name="churn_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                on_click="ignore"
            )




