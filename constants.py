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
    "Store": {"store", "rooftop", "location", "franchise", "dealer name", "store name", "rooftop name", "location name", "dealership", "dealer", "rooftop id", "dlr", "dealership name"},
    "Deal_Number": {"deal_number", "deal#", "dealno", "deal", "ro", "rono", "ro#", "stockdeal", "deal id", "deal num", "deal number", "doc#", "doc no"},
    "CustomerID": {"cust_no", "customerid", "custid", "cid", "dmsid", "customer_number", "customerno", "accountid", "account id", "acctid", "clientid"},
    "First_Name": {"firstname", "givenname", "first name", "buyer first", "primary first"},
    "Last_Name": {"lastname", "surname", "familyname", "last name", "buyer last", "primary last"},
    "FullName": {"full name", "fullname", "customer name", "customername", "contact name", "contactname", "buyer name", "primary name"},
    # Co-buyer fields (kept internal; used for exclusion and not output)
    "Co_First_Name": {"co first", "co_first", "co-buyer first", "cobuyer first", "co buyer first", "cofirstname", "co first name", "co-buyer firstname"},
    "Co_Last_Name": {"co last", "co_last", "co-buyer last", "cobuyer last", "co buyer last", "colastname", "co last name", "co-buyer lastname"},
    "Co_FullName": {"co buyer", "co-buyer", "cobuyer", "co name", "co fullname", "co full name", "co-buyer name", "co-buyer fullname"},
    "Email": {"email", "emailaddress", "e-mail"},
    "Home_Phone": {"home", "homephone", "home phone", "residence phone"},
    "Mobile_Phone": {"phonecell", "mobile", "cell", "cellphone", "mobilephone", "mobile phone", "cell phone"},
    "Work_Phone": {"work", "workphone", "businessphone", "work phone", "office phone", "business phone"},
    "Phone2": {"phone2", "altphone", "secondaryphone", "otherphone", "alternate phone", "alt phone", "phone", "phone number", "telephone", "tel"},
    # Phone area codes (used only for preprocessing merge)
    "Home_AreaCode": {"home area code", "home area", "home ac", "homeareacode", "res area", "res area code"},
    "Mobile_AreaCode": {"mobile area code", "cell area code", "cell area", "mobile area", "cell ac"},
    "Work_AreaCode": {"work area code", "office area code", "work area", "office area", "business area"},
    "Phone2_AreaCode": {"alt area code", "alternate area code", "secondary area code", "other area code"},
    "AreaCode": {"area code", "areacode", "area", "ac"},
    "Address1": {
        "street", "address", "address1", "addr1", "streetaddress",
        "address line 1", "addressline1", "customer address", "customeraddress",
        "mailing address", "street addr", "primary address", "residential address"
    },
    "Address2": {
        "address2", "addr2", "address line 2", "addressline2",
        "suite", "ste", "unit", "apt", "apartment", "po box", "p.o. box", "pobox",
        "building", "bldg", "floor", "fl", "room", "rm"
    },
    "City": {"city", "town", "municipality", "locality", "city name", "csz city"},
    "State": {"state", "st", "province", "region", "state/province", "state code", "csz state"},
    "Zip": {"zip", "zip5", "zipcode", "postalcode", "postcode", "zip code", "postal code", "csz zip"},
    "VIN": {"vin", "vehicleid", "vehicle id", "vin number"},
    "Make": {"make", "manufacturer", "brand", "oem"},
    "Model": {"model", "vehiclemodel", "series", "trim", "model name"},
    "Year": {"year", "modelyear", "vehicleyear", "model year"},
    "DeliveryDate": {"del_date", "deliverydate", "sold_date", "solddate", "sale_date", "saledate", "date", "delivery", "delivered"},
    "Delivery_Miles": {"del_miles", "deliverymiles", "delivery_mileage", "miles_at_delivery"},
    "Distance": {"dist", "distance", "milesaway", "distance_to_dealer", "customerdistance"},
    # Vehicle_Condition should map to New/Used status; avoid 'vehicle type'
    "Vehicle_Condition": {"newused", "condition", "stocktype", "nu", "n/u"},
    "Mileage": {"mileage", "odometer", "odo", "currentmileage", "current_odometer", "miles", "odometer reading"},
    "Term": {"term", "termmonths", "financeterm", "leaseterm", "term months", "months term"},
}

# Negative keywords to de-prioritize misleading header matches for specific canonicals
NEGATIVE_KEYWORDS = {
    "Store": {"id", "number", "no", "code", "fee", "fees", "charge", "charges", "dv"},
    "Deal_Number": {"type", "status", "fee", "fees", "amount", "total", "balance"},
    "Address1": {"line 2", "address2", "addr2", "bank", "billing", "mailing", "finance", "lien"},
    "Address2": {"line 1", "address1", "addr1"},
    "City": {"co-buyer", "cobuyer", "co buyer", "bank", "billing", "mailing", "shipping", "lien", "finance", "type", "file", "status", "code", "first", "last", "name", "full name", "fullname", "model", "series", "trim", "vin", "stock", "body"},
    "VIN": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    # Penalize VIN mapping on obviously wrong headers
    "VIN": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2", "make", "model", "type", "file", "explosion"},
    "Make": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    "Model": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2"},
    "Year": {"trade", "trade-in", "tradein", "t1", "t2", "trade 1", "trade 2", "model", "series", "trim"},
    "Vehicle_Condition": {"type", "vehicle type", "body", "bodystyle", "style"},
    # Prevent names mapping to date/activity columns
    "First_Name": {"date", "entry", "activity", "pay", "payment", "due"},
    "Last_Name": {"date", "entry", "activity", "pay", "payment", "due"},
    "FullName": {"date", "entry", "activity", "pay", "payment", "due"},
    # Prevent co-buyer columns from being mapped to primary buyer fields
    "First_Name": {"co", "co buyer", "co-buyer", "cobuyer", "co first", "co_first", "cofirst"},
    "Last_Name": {"co", "co buyer", "co-buyer", "cobuyer", "co last", "co_last", "colast"},
    "FullName": {"co", "co buyer", "co-buyer", "cobuyer", "co name", "co fullname", "co full name"},
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
    "delete_out_of_state": False,
    "home_state": "CA",       # default; can be changed later
    # Enable model year window 2013..2024 inclusive
    "model_year_filter": {"enabled": True, "min_year": 2013, "max_year": 2024},
    "delivery_age_filter": {"enabled": True, "months": 18},
    "distance_filter": {"enabled": True, "max_miles": 100},
    "exclude_corporate": True,
    "exclude_cobuyers": False,  # disabled by default for Milestone 1
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


