import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="VoltWise | Smart Electricity",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. PROFESSIONAL CSS DESIGN
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #111C2E;
    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] * {
    font-size: 15px;
}

h1 {
    color: #FFFFFF !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

h2 {
    color: #FFFFFF !important;
    font-size: 27px !important;
}

h3 {
    color: #FFFFFF !important;
    font-size: 21px !important;
}

p, label, li {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

[data-testid="stMetric"] {
    background: #17243A;
    border: 1px solid #2A3D59;
    padding: 22px;
    border-radius: 15px;
    min-height: 125px;
}

[data-testid="stMetricLabel"] {
    color: #B6C6D9 !important;
    font-size: 15px !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 29px !important;
    font-weight: 800 !important;
}

.stButton button {
    background: #19B995;
    color: #061B17;
    border-radius: 10px;
    border: none;
    font-size: 15px;
    font-weight: 700;
    min-height: 45px;
}

.stDownloadButton button {
    background: #19B995;
    color: #061B17;
    border-radius: 10px;
    border: none;
    font-size: 15px;
    font-weight: 700;
    min-height: 48px;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #45D8B5;
    color: #061B17;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2A3D59;
    border-radius: 10px;
}

hr {
    border-color: #2A3D59;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD DATASET AND MODEL
# ==========================================================

@st.cache_data
def load_dataset():

    return pd.read_csv("electricity_bill_dataset.csv")


@st.cache_resource
def load_model():

    return joblib.load("electricity_model.pkl")


try:

    data = load_dataset()

    model = load_model()

except FileNotFoundError as error:

    st.error(
        f"File not found: {error.filename}. "
        "Keep the dataset, model and app.py in the same folder."
    )

    st.stop()

except Exception as error:

    st.error(f"Unable to load project files: {error}")

    st.stop()


# ==========================================================
# 4. VALIDATE DATASET
# ==========================================================

required_columns = [
    "Fan",
    "Refrigerator",
    "AirConditioner",
    "Television",
    "Monitor",
    "MotorPump",
    "Month",
    "City",
    "Company",
    "MonthlyHours",
    "TariffRate",
    "ElectricityBill"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:

    st.error(f"Missing dataset columns: {missing_columns}")

    st.stop()


# ==========================================================
# 5. SESSION STATE
# ==========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ==========================================================
# 6. SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("⚡ VOLTWISE")

    st.caption("INTELLIGENT ENERGY MANAGEMENT")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Bill Prediction",
            "Monthly Analysis",
            "Prediction History",
            "Project Information"
        ]
    )

    st.divider()

    st.subheader("Household Appliances")

    fan = st.number_input(
        "Number of Fans",
        0, 10, 3
    )

    refrigerator = st.number_input(
        "Refrigerators",
        0, 3, 1
    )

    ac = st.number_input(
        "Air Conditioners",
        0, 5, 1
    )

    tv = st.number_input(
        "Televisions",
        0, 5, 1
    )

    monitor = st.number_input(
        "Monitors",
        0, 5, 1
    )

    pump = st.number_input(
        "Motor Pumps",
        0, 3, 0
    )

    st.divider()

    st.subheader("Electricity Details")

    months = sorted(
        data["Month"].dropna().unique().tolist(),
        key=str
    )

    cities = sorted(
        data["City"].dropna().unique().tolist(),
        key=str
    )

    companies = sorted(
        data["Company"].dropna().unique().tolist(),
        key=str
    )

    month = st.selectbox(
        "Select Month",
        months
    )

    city = st.selectbox(
        "Select City",
        cities
    )

    company = st.selectbox(
        "Electricity Company",
        companies
    )

    monthly_hours = st.number_input(
        "Monthly Usage Hours",
        min_value=1,
        max_value=1000,
        value=200
    )

    tariff_rate = st.number_input(
        "Tariff Rate (₹)",
        min_value=0.1,
        max_value=100.0,
        value=8.0,
        step=0.1
    )


# ==========================================================
# 7. PREPARE INPUT DATA
# ==========================================================

input_data = pd.DataFrame({

    "Fan": [fan],

    "Refrigerator": [refrigerator],

    "AirConditioner": [ac],

    "Television": [tv],

    "Monitor": [monitor],

    "MotorPump": [pump],

    "Month": [month],

    "City": [city],

    "Company": [company],

    "MonthlyHours": [monthly_hours],

    "TariffRate": [tariff_rate]

})


# ==========================================================
# 8. MACHINE LEARNING PREDICTION
# ==========================================================

current_bill = float(
    model.predict(input_data)[0]
)

# Avoid showing a negative predicted bill.

current_bill = max(0, current_bill)


# ==========================================================
# 9. COMMON DISPLAY FUNCTIONS
# ==========================================================

def show_header(title, description):

    st.caption("⚡ VOLTWISE  |  ENERGY INTELLIGENCE")

    st.title(title)

    st.markdown(description)

    st.divider()


def show_section(title):

    st.subheader(title)


def create_chart():

    fig, ax = plt.subplots(figsize=(8, 4))

    fig.patch.set_facecolor("#17243A")

    ax.set_facecolor("#17243A")

    ax.tick_params(
        colors="#E5EDF7",
        labelsize=11
    )

    ax.xaxis.label.set_color("#E5EDF7")

    ax.yaxis.label.set_color("#E5EDF7")

    ax.title.set_color("#FFFFFF")

    for spine in ax.spines.values():

        spine.set_color("#52657E")

    ax.grid(
        axis="y",
        color="#52657E",
        alpha=0.35,
        linestyle="--"
    )

    return fig, ax


# ==========================================================
# 10. OPTIMIZATION SCENARIOS
# ==========================================================

def calculate_scenarios():

    results = []

    for reduction in [5, 10, 15, 20]:

        reduced_hours = max(
            1,
            round(
                monthly_hours * (1 - reduction / 100)
            )
        )

        modified_input = input_data.copy()

        modified_input["MonthlyHours"] = reduced_hours

        predicted_bill = float(
            model.predict(modified_input)[0]
        )

        predicted_bill = max(0, predicted_bill)

        savings = current_bill - predicted_bill

        results.append({

            "Usage Reduction": f"{reduction}%",

            "Usage Hours": reduced_hours,

            "Predicted Bill (₹)": round(
                predicted_bill, 2
            ),

            "Estimated Savings (₹)": round(
                savings, 2
            )

        })

    return pd.DataFrame(results)


scenarios = calculate_scenarios()


# ==========================================================
# 11. PDF ELECTRICITY BILL GENERATOR
# ==========================================================

def generate_electricity_bill(
    reduction,
    optimized_hours,
    optimized_bill,
    savings
):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    navy = colors.HexColor("#142B46")

    green = colors.HexColor("#16A889")

    light = colors.HexColor("#EDF4F8")

    grey = colors.HexColor("#64748B")

    white = colors.white

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BillTitle",
        fontName="Helvetica-Bold",
        fontSize=27,
        textColor=white,
        leading=32
    )

    subtitle_style = ParagraphStyle(
        "BillSubtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#D0E1E9"),
        leading=14
    )

    heading_style = ParagraphStyle(
        "BillHeading",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=navy,
        leading=16
    )

    normal_style = ParagraphStyle(
        "BillNormal",
        fontName="Helvetica",
        fontSize=9,
        textColor=grey,
        leading=13
    )

    amount_style = ParagraphStyle(
        "BillAmount",
        fontName="Helvetica-Bold",
        fontSize=25,
        textColor=white,
        alignment=TA_CENTER
    )

    center_style = ParagraphStyle(
        "BillCenter",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=white,
        alignment=TA_CENTER
    )

    elements = []

    # HEADER

    header = Table(
        [
            [
                Paragraph("VOLTWISE", title_style),

                Paragraph(
                    "ELECTRICITY BILL<br/>STATEMENT",
                    center_style
                )
            ],
            [
                Paragraph(
                    "INTELLIGENT ELECTRICITY CONSUMPTION OPTIMIZATION",
                    subtitle_style
                ),

                Paragraph(
                    "MACHINE LEARNING PROJECT",
                    subtitle_style
                )
            ]
        ],
        colWidths=[340, 160],
        rowHeights=[45, 28]
    )

    header.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, -1), navy),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 14),

        ("RIGHTPADDING", (0, 0), (-1, -1), 14),

    ]))

    elements.append(header)

    elements.append(Spacer(1, 20))

    # BILL DETAILS

    elements.append(
        Paragraph(
            "ELECTRICITY BILL DETAILS",
            heading_style
        )
    )

    elements.append(Spacer(1, 10))

    bill_number = "VW-" + datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    details = [

        [
            "Bill Number",
            bill_number,
            "Bill Date",
            datetime.now().strftime("%d-%m-%Y")
        ],

        [
            "Billing Month",
            str(month),
            "City",
            str(city)
        ],

        [
            "Consumer",
            "Household Customer",
            "Provider",
            str(company)
        ]

    ]

    details_table = Table(
        details,
        colWidths=[90, 150, 75, 185],
        rowHeights=30
    )

    details_table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, -1), light),

        ("GRID", (0, 0), (-1, -1), 0.5,
         colors.HexColor("#D5E0EA")),

        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),

        ("FONTSIZE", (0, 0), (-1, -1), 8),

        ("TEXTCOLOR", (0, 0), (-1, -1), grey),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),

    ]))

    elements.append(details_table)

    elements.append(Spacer(1, 20))

    # APPLIANCE DETAILS

    elements.append(
        Paragraph(
            "01 | APPLIANCE CONSUMPTION DETAILS",
            heading_style
        )
    )

    elements.append(Spacer(1, 9))

    consumption = [

        ["APPLIANCE / USAGE", "QUANTITY / VALUE"],

        ["Fans", str(fan)],

        ["Refrigerators", str(refrigerator)],

        ["Air Conditioners", str(ac)],

        ["Televisions", str(tv)],

        ["Monitors", str(monitor)],

        ["Motor Pumps", str(pump)],

        ["Monthly Usage Hours", f"{monthly_hours} Hours"],

        ["Tariff Rate", f"Rs. {tariff_rate:.2f}"]

    ]

    consumption_table = Table(
        consumption,
        colWidths=[325, 175],
        rowHeights=25
    )

    consumption_table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), navy),

        ("TEXTCOLOR", (0, 0), (-1, 0), white),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),

        ("FONTSIZE", (0, 0), (-1, -1), 9),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [white, light]),

        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#D5E0EA")),

        ("ALIGN", (1, 0), (1, -1), "CENTER"),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),

    ]))

    elements.append(consumption_table)

    elements.append(Spacer(1, 20))

    # BILL SUMMARY

    elements.append(
        Paragraph(
            "02 | BILL SUMMARY",
            heading_style
        )
    )

    elements.append(Spacer(1, 9))

    amount_box = Table(

        [

            [
                Paragraph(
                    "TOTAL PREDICTED ELECTRICITY BILL",
                    center_style
                )
            ],

            [
                Paragraph(
                    f"Rs. {current_bill:,.2f}",
                    amount_style
                )
            ]

        ],

        colWidths=[500],

        rowHeights=[30, 48]

    )

    amount_box.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, -1), green),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

    ]))

    elements.append(amount_box)

    elements.append(Spacer(1, 18))

    # OPTIMIZATION DETAILS

    elements.append(
        Paragraph(
            "03 | CONSUMPTION OPTIMIZATION",
            heading_style
        )
    )

    elements.append(Spacer(1, 9))

    optimization = [

        ["DESCRIPTION", "RESULT"],

        ["Usage Reduction", f"{reduction}%"],

        ["Optimized Usage", f"{optimized_hours} Hours"],

        ["Optimized Bill", f"Rs. {optimized_bill:,.2f}"],

        ["Estimated Savings", f"Rs. {savings:,.2f}"]

    ]

    optimization_table = Table(
        optimization,
        colWidths=[325, 175],
        rowHeights=27
    )

    optimization_table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), navy),

        ("TEXTCOLOR", (0, 0), (-1, 0), white),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("FONTSIZE", (0, 0), (-1, -1), 9),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [white, light]),

        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#D5E0EA")),

        ("ALIGN", (1, 0), (1, -1), "CENTER"),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),

        ("BACKGROUND", (0, -1), (-1, -1),
         colors.HexColor("#D8F5EA")),

        ("TEXTCOLOR", (0, -1), (-1, -1), navy),

        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),

    ]))

    elements.append(optimization_table)

    elements.append(Spacer(1, 20))

    # FOOTER

    footer_style = ParagraphStyle(
        "Footer",
        fontName="Helvetica",
        fontSize=8,
        leading=13,
        textColor=grey,
        alignment=TA_CENTER
    )

    footer = Table(

        [[
            Paragraph(
                "<b>SMARTER ENERGY. BETTER TOMORROW.</b><br/><br/>"
                "Thank you for using VoltWise.<br/>"
                "This is a machine-learning-generated estimated "
                "bill for academic demonstration. It is not an "
                "official electricity utility invoice. Actual "
                "electricity bills may differ.",
                footer_style
            )
        ]],

        colWidths=[500]

    )

    footer.setStyle(TableStyle([

        ("LINEABOVE", (0, 0), (-1, -1), 1, green),

        ("TOPPADDING", (0, 0), (-1, -1), 12),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),

    ]))

    elements.append(footer)

    pdf.build(elements)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================================
# 12. DASHBOARD PAGE
# ==========================================================

if page == "Dashboard":

    show_header(
        "Smart Electricity Dashboard",
        "Monitor predicted bills, understand household "
        "usage and explore electricity-saving opportunities."
    )

    show_section("Energy Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Monthly Bill",
        f"₹{current_bill:,.2f}"
    )

    col2.metric(
        "Monthly Usage",
        f"{monthly_hours} Hours"
    )

    col3.metric(
        "Dataset Records",
        f"{len(data):,}"
    )

    col4.metric(
        "ML Algorithm",
        "Random Forest"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        show_section("Predicted Electricity Cost")

        fig, ax = create_chart()

        ax.bar(
            ["Predicted Bill"],
            [current_bill]
        )

        ax.set_ylabel("Electricity Bill (₹)")

        ax.set_title("Estimated Monthly Bill")

        st.pyplot(fig)

        plt.close(fig)

    with right:

        show_section("Platform Features")

        st.info(
            "**01. Bill Prediction**\n\n"
            "Estimate electricity bills using household "
            "appliances and electricity details."
        )

        st.info(
            "**02. Consumption Optimization**\n\n"
            "Compare bills under different usage-reduction scenarios."
        )

        st.info(
            "**03. Data Analysis**\n\n"
            "Explore electricity trends and dataset statistics."
        )

    st.divider()

    show_section("Potential Savings")

    ten_percent_saving = scenarios.iloc[1][
        "Estimated Savings (₹)"
    ]

    st.metric(
        "Estimated Savings at 10% Usage Reduction",
        f"₹{ten_percent_saving:,.2f}"
    )

    st.caption(
        "Change the household details in the sidebar "
        "to update the prediction."
    )


# ==========================================================
# 13. BILL PREDICTION PAGE
# ==========================================================

elif page == "Bill Prediction":

    show_header(
        "Electricity Bill Prediction",
        "Predict your electricity bill and explore "
        "consumption optimization opportunities."
    )

    show_section("Current Bill Estimate")

    col1, col2 = st.columns(2)

    col1.metric(
        "Current Predicted Bill",
        f"₹{current_bill:,.2f}"
    )

    col2.metric(
        "Monthly Usage",
        f"{monthly_hours} Hours"
    )

    st.divider()

    show_section("Consumption Optimization")

    reduction = st.select_slider(
        "Choose Usage Reduction",
        options=[5, 10, 15, 20],
        value=10
    )

    optimized_hours = max(
        1,
        round(
            monthly_hours * (1 - reduction / 100)
        )
    )

    optimized_input = input_data.copy()

    optimized_input["MonthlyHours"] = optimized_hours

    optimized_bill = float(
        model.predict(optimized_input)[0]
    )

    optimized_bill = max(0, optimized_bill)

    savings = current_bill - optimized_bill

    savings_percentage = (
        savings / current_bill * 100
        if current_bill > 0 else 0
    )

    a, b, c = st.columns(3)

    a.metric(
        "Optimized Bill",
        f"₹{optimized_bill:,.2f}"
    )

    b.metric(
        "Estimated Savings",
        f"₹{savings:,.2f}",
        f"{savings_percentage:.1f}%"
    )

    c.metric(
        "Reduced Usage",
        f"{optimized_hours} Hours"
    )

    st.divider()

    show_section("Optimization Scenario Comparison")

    st.dataframe(
        scenarios,
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:

        show_section("Current vs Optimized Bill")

        fig, ax = create_chart()

        ax.bar(
            ["Current", "Optimized"],
            [current_bill, optimized_bill]
        )

        ax.set_ylabel("Bill Amount (₹)")

        st.pyplot(fig)

        plt.close(fig)

    with right:

        show_section("Estimated Savings")

        fig, ax = create_chart()

        ax.bar(
            scenarios["Usage Reduction"],
            scenarios["Estimated Savings (₹)"]
        )

        ax.set_ylabel("Savings (₹)")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    show_section("Personalized Energy-Saving Tips")

    tips = []

    if ac > 0:
        tips.append(
            "Use air conditioners at a moderate temperature."
        )

    if fan > 0:
        tips.append(
            "Switch off fans when rooms are unoccupied."
        )

    if refrigerator > 0:
        tips.append(
            "Avoid keeping refrigerator doors open unnecessarily."
        )

    if pump > 0:
        tips.append(
            "Operate motor pumps only when required."
        )

    if tv > 0:
        tips.append(
            "Switch off televisions when not in use."
        )

    if monitor > 0:
        tips.append(
            "Turn off monitors when computers are not being used."
        )

    for tip in tips:

        st.markdown(f"✓ {tip}")

    st.divider()

    # PDF DOWNLOAD

    show_section("Download Your Electricity Bill")

    st.info(
        "Download a professionally formatted A4 PDF "
        "containing your electricity details, predicted bill "
        "and estimated optimization savings."
    )

    pdf_bill = generate_electricity_bill(
        reduction,
        optimized_hours,
        optimized_bill,
        savings
    )

    st.download_button(
        label="📄 DOWNLOAD ELECTRICITY BILL (PDF)",
        data=pdf_bill,
        file_name="VoltWise_Electricity_Bill.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.divider()

    if st.button("Save Prediction"):

        st.session_state.history.insert(0, {

            "Date": datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            ),

            "Current Bill": round(
                current_bill, 2
            ),

            "Reduction": f"{reduction}%",

            "Optimized Bill": round(
                optimized_bill, 2
            ),

            "Savings": round(
                savings, 2
            )

        })

        st.success("Prediction saved successfully!")


# ==========================================================
# 14. MONTHLY ANALYSIS PAGE
# ==========================================================

elif page == "Monthly Analysis":

    show_header(
        "Monthly Electricity Analysis",
        "Explore electricity consumption data "
        "using visualizations and statistical summaries."
    )

    monthly = (
        data.groupby("Month")["ElectricityBill"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Average Bill (₹)",
        "Median Bill (₹)",
        "Records"
    ]

    monthly = monthly.sort_values(
        "Month",
        key=lambda s: s.astype(str)
    )

    a, b, c = st.columns(3)

    a.metric(
        "Total Records",
        f"{len(data):,}"
    )

    b.metric(
        "Average Bill",
        f"₹{data['ElectricityBill'].mean():,.2f}"
    )

    c.metric(
        "Highest Bill",
        f"₹{data['ElectricityBill'].max():,.2f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        show_section("Average Bill by Month")

        fig, ax = create_chart()

        ax.plot(
            monthly["Month"].astype(str),
            monthly["Average Bill (₹)"],
            marker="o",
            linewidth=2.5
        )

        ax.set_ylabel("Average Bill (₹)")

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)

        plt.close(fig)

    with right:

        show_section("Electricity Bill Distribution")

        fig, ax = create_chart()

        ax.hist(
            data["ElectricityBill"].dropna(),
            bins=30
        )

        ax.set_xlabel("Electricity Bill (₹)")

        ax.set_ylabel("Number of Records")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    show_section("Monthly Statistics")

    st.dataframe(
        monthly,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Analysis is based on the supplied dataset, "
        "not live electricity meter readings."
    )


# ==========================================================
# 15. PREDICTION HISTORY PAGE
# ==========================================================

elif page == "Prediction History":

    show_header(
        "Prediction History",
        "Review electricity predictions saved "
        "during your current application session."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Download History CSV",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="Prediction_History.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No saved predictions yet. "
            "Visit Bill Prediction and select Save Prediction."
        )

    st.caption(
        "Prediction history is temporary and lasts "
        "only for the current application session."
    )


# ==========================================================
# 16. PROJECT INFORMATION PAGE
# ==========================================================

elif page == "Project Information":

    show_header(
        "Project Information",
        "Technical overview of the Intelligent Electricity "
        "Consumption Optimization System Using Machine Learning."
    )

    show_section("Model Configuration")

    a, b, c = st.columns(3)

    a.metric("Algorithm", "Random Forest")

    b.metric("Dataset Records", f"{len(data):,}")

    c.metric("Target Variable", "Electricity Bill")

    st.divider()

    show_section("Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**PYTHON**\n\n"
            "Core programming language used "
            "for development and data processing."
        )

    with col2:

        st.info(
            "**SCIKIT-LEARN**\n\n"
            "Used to build and train the "
            "Random Forest regression model."
        )

    with col3:

        st.info(
            "**STREAMLIT**\n\n"
            "Used to create the interactive "
            "web application."
        )

    st.divider()

    show_section("System Workflow")

    workflow = [

        (
            "01",
            "Dataset Collection",
            "Collect electricity consumption and bill records."
        ),

        (
            "02",
            "Data Preprocessing",
            "Prepare numerical and categorical features."
        ),

        (
            "03",
            "Model Training",
            "Train the Random Forest regression pipeline."
        ),

        (
            "04",
            "Bill Prediction",
            "Predict electricity bills from user inputs."
        ),

        (
            "05",
            "Consumption Optimization",
            "Simulate reduced usage-hour scenarios."
        ),

        (
            "06",
            "Result Analysis",
            "Compare estimated bills and potential savings."
        )

    ]

    for i in range(0, len(workflow), 2):

        left, right = st.columns(2)

        for col, item in zip(
            [left, right],
            workflow[i:i + 2]
        ):

            with col:

                st.markdown(f"### STEP {item[0]}")

                st.subheader(item[1])

                st.write(item[2])

                st.divider()

    st.warning(
        "This is an academic prototype. Optimization scenarios "
        "simulate reduced usage hours. The application does not "
        "directly control appliances or measure live electricity."
    )


# ==========================================================
# 17. FOOTER
# ==========================================================

st.divider()

st.markdown("### ⚡ VOLTWISE")

st.caption(
    "INTELLIGENT ELECTRICITY CONSUMPTION OPTIMIZATION"
)

st.caption(
    "Machine Learning Academic Project"
)

st.caption(
    "Predictions are estimates and may differ "
    "from actual electricity bills."
)