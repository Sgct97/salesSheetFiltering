from __future__ import annotations

# ===== Canonical output order (locked) =====
CANONICAL_OUTPUT_ORDER = [
    "Store",
    "Deal_Number",
    "CustomerID",
    "Last_Name",
    "First_Name",
    "FullName",
    "Email",
    "Home_Phone",
    "Mobile_Phone",
    "Work_Phone",
    "Phone2",
    "Address1",
    "Address2",
    "City",
    "State",
    "Zip",
    "VIN",
    "Make",
    "Model",
    "Year",
    "DeliveryDate",
    "Delivery_Miles",
    "Distance",
    "Vehicle_Condition",  # New/Used
    "Mileage",            # Odometer
    "Term",
]

# ===== Synonyms (normalized, case/punct-insensitive used in detection) =====
SYNONYMS = {
    "Store": {"store", "rooftop", "location", "franchise", "dealer name", "store name", "rooftop name", "location name", "dealership"},
    "Deal_Number": {"deal_number", "deal#", "dealno", "deal", "ro", "rono", "ro#", "stockdeal"},
    "CustomerID": {"cust_no", "customerid", "custid", "cid", "dmsid", "customer_number", "customerno", "accountid"},
    "First_Name": {"first", "firstname", "givenname"},
    "Last_Name": {"last", "lastname", "surname", "familyname"},
    "FullName": {"full name", "fullname", "name", "customer name", "customername", "contact name", "contactname"},
    "Email": {"email", "emailaddress", "e-mail"},
    "Home_Phone": {"home", "homephone"},
    "Mobile_Phone": {"phonecell", "mobile", "cell", "cellphone", "mobilephone"},
    "Work_Phone": {"work", "workphone", "businessphone"},
    "Phone2": {"phone2", "altphone", "secondaryphone", "otherphone"},
    "Address1": {
        "street", "address", "address1", "addr1", "streetaddress",
        "address line 1", "addressline1", "customer address", "customeraddress"
    },
    "Address2": {
        "address2", "addr2", "address line 2", "addressline2",
        "suite", "ste", "unit", "apt", "apartment", "po box", "p.o. box", "pobox"
    },
    "City": {"city", "town", "municipality", "locality"},
    "State": {"state", "st", "province", "region"},
    "Zip": {"zip", "zip5", "zipcode", "postalcode", "postcode"},
    "VIN": {"vin", "vehicleid"},
    "Make": {"make", "manufacturer"},
    "Model": {"model", "vehiclemodel", "series", "trim"},
    "Year": {"year", "modelyear", "vehicleyear"},
    "DeliveryDate": {"del_date", "deliverydate", "sold_date", "solddate", "sale_date", "saledate", "date"},
    "Delivery_Miles": {"del_miles", "deliverymiles", "delivery_mileage", "miles_at_delivery"},
    "Distance": {"dist", "distance", "milesaway", "distance_to_dealer", "customerdistance"},
    "Vehicle_Condition": {"newused", "condition", "stocktype", "vehicletype", "nu", "n/u"},
    "Mileage": {"mileage", "odometer", "odo", "currentmileage", "current_odometer"},
    "Term": {"term", "termmonths", "financeterm", "leaseterm"},
}

# Negative keywords to de-prioritize misleading header matches for specific canonicals
NEGATIVE_KEYWORDS = {
    "Store": {"id", "number", "no", "code", "fee", "fees", "charge", "charges", "dv"},
    "Deal_Number": {"type", "status", "fee", "fees", "amount", "total", "balance"},
    "Address1": {"line 2", "address2", "addr2", "bank", "billing", "mailing", "finance", "lien"},
    "Address2": {"line 1", "address1", "addr1"},
    "City": {"co-buyer", "cobuyer", "co buyer", "bank", "billing", "mailing", "shipping", "lien", "finance"},
    # Penalize generic meta columns for City mapping
    "City": {"co-buyer", "cobuyer", "co buyer", "bank", "billing", "mailing", "shipping", "lien", "finance", "type", "file", "status", "code"},
    "VIN": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    # Penalize VIN mapping on obviously wrong headers
    "VIN": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2", "make", "model", "type", "file", "explosion"},
    "Make": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    "Model": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    "Year": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
}

# Positive keywords to boost likely matches
POSITIVE_KEYWORDS = {
    "Address1": {"line 1", "address1", "addr1", "street"},
    "Address2": {"line 2", "address2", "addr2", "suite", "unit", "apt", "apartment", "po box", "p.o. box", "pobox"},
}

# Distinct VIN explosion source (not the VIN field)
VIN_EXPLOSION_SYNONYMS = {
    "vin explosion", "vin list", "vin_list", "vins", "vin(s)",
    "multi vin", "multivin", "vin batch", "vin csv", "vin pipe", "vin delim", "vin_split", "vins_found"
}
VIN_EXPLOSION_DELIMITERS = [",", ";", "|", "/", " ", "\t"]

# ===== Regex/value patterns =====
VIN_REGEX = r"(?i)\b(?![IOQ])[A-HJ-NPR-Z0-9]{17}\b"
EMAIL_REGEX = r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
US_PHONE_REGEX = r"(?x)\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
ZIP5_REGEX = r"\b\d{5}(?:-\d{4})?\b"

# ===== States (2-letter) quick map for normalization =====
US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

# ===== Corporate/Dealer/Auction exclusion lexicons =====
EXCLUDE_BRANDS = {
    "MANHEIM", "ADESA", "CARMAX", "AUTONATION", "LITHIA", "PENSKE", "SONIC", "GROUP 1", "CARVANA",
    "ENTERPRISE", "HERTZ", "AVIS", "BUDGET", "COPART", "IAA", "VROOM"
}
EXCLUDE_OEMS = {
    "TOYOTA","HONDA","CHEVROLET","CHEVY","FORD","NISSAN","KIA","HYUNDAI","CHRYSLER","DODGE",
    "JEEP","RAM","VOLKSWAGEN","VW","AUDI","BMW","MERCEDES","MERCEDES-BENZ","LEXUS","ACURA",
    "INFINITI","GMC","BUICK","CADILLAC","SUBARU","MAZDA","VOLVO","PORSCHE"
}
EXCLUDE_KEYWORDS = {
    "AUTO","MOTORS","SALES","SERVICE","PARTS","BODY","FLEET","WHOLESALE","AUCTION","DEALER","ROOFTOP","GROUP","HOLDINGS"
}
CORPORATE_SUFFIXES = {"INC","LLC","LLP","PLC","CORP","CO","COMPANY","LTD","TRUST","FOUNDATION","ASSOCIATION"}

# ===== Name vs Company heuristics =====
FULLNAME_NEGATIVE_KEYWORDS = {"company", "business", "dealer", "rooftop", "auction"}
PERSON_TITLE_CASE_REQUIRED = True  # for strong person-like override

# ===== Delivery date precedence =====
DELIVERYDATE_PRECEDENCE = ["del_date", "sold_date", "sale_date", "deliverydate", "date"]

# ===== Milestone 1 presets (fixed) =====
PRESETS = {
    "delete_duplicates": True,
    "vin_explosion": True,  # only if a VIN explosion source column is present
    "address_present": True,  # Require Address1 + City + State + Zip (PO BOX counts)
    "name_present": False,     # Disabled: do not exclude rows for name presence in Milestone 1
    "delete_out_of_state": True,
    "home_state": "WA",       # default; can be changed later
    "model_year_filter": {"enabled": False, "operator": "newer", "year": None},
    "delivery_age_filter": {"enabled": True, "months": 18},
    "distance_filter": {"enabled": True, "max_miles": 100},
    "exclude_corporate": True,
    "preserve_all_columns": False,
    "po_box_counts_as_address": True,
}

# ===== Scoring thresholds for header/value matcher =====
FUZZY_STRONG = 90
FUZZY_CANDIDATE = 80
VALUE_STRONG_SCORE = 3
BRAND_SCORE = 3
OEM_SCORE = 2
AUTO_KEYWORD_SCORE = 2
ALLCAPS_MULTITOKEN_SCORE = 2
COMPANY_PRESENT_SCORE = 1
PERSON_OVERRIDE_SCORE = -3
EXCLUDE_THRESHOLD = 3


