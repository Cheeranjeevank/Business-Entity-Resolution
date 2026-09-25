import pandas as pd
import unicodedata
import re

# Dictionaries for standardization
LEGAL_SUFFIXES = {
    r'\b(ltd|limited)\b': 'ltd',
    r'\b(pvt|private)\b': 'pvt',
    r'\b(inc|incorporated)\b': 'inc',
    r'\b(corp|corporation)\b': 'corp',
    r'\b(llc)\b': 'llc',
    r'\b(co|company)\b': 'co'
}

ADDRESS_ABBREV = {
    r'\b(rd|road)\b': 'rd',
    r'\b(st|street)\b': 'st',
    r'\b(ave|avenue)\b': 'ave',
    r'\b(blvd|boulevard)\b': 'blvd',
    r'\b(dr|drive)\b': 'dr',
    r'\b(ln|lane)\b': 'ln',
    r'\b(pl|place)\b': 'pl',
    r'\b(apt|apartment)\b': 'apt',
    r'\b(ste|suite)\b': 'ste',
    r'\b(bldg|building)\b': 'bldg',
    r'\b(fl|floor)\b': 'fl'
}

def unicode_normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Normalize unicode (removes accents/diacritics)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def normalize_business_name(name):
    name = unicode_normalize(name)
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Standardize legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES.items():
        name = re.sub(pattern, replacement, name)
        
    # Standardize spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_business_address(address):
    address = unicode_normalize(address)
    
    # Remove punctuation
    address = re.sub(r'[^\w\s]', ' ', address)
    
    # Standardize street abbreviations
    for pattern, replacement in ADDRESS_ABBREV.items():
        address = re.sub(pattern, replacement, address)
        
    # Standardize spaces
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def extract_postal_code(address):
    # Extracts US (5 digit) or India (6 digit) postal codes
    if pd.isna(address):
        return ""
    
    # India PIN codes (6 digits)
    india_pin = re.search(r'\b\d{6}\b', str(address))
    if india_pin:
        return india_pin.group(0)
        
    # US Zip codes (5 digits)
    us_zip = re.search(r'\b\d{5}\b', str(address))
    if us_zip:
        return us_zip.group(0)
        
    return ""

def extract_numbers(address):
    # Extracts all numbers (could be house numbers, unit numbers, etc.)
    if pd.isna(address):
        return ""
    numbers = re.findall(r'\b\d+\b', str(address))
    return " ".join(numbers)

def normalize_dataframe(df):
    """
    Applies all normalizations and feature extractions to a dataframe while keeping original columns.
    """
    print("Normalizing business names...")
    df['norm_name'] = df['business_name'].apply(normalize_business_name)
    
    print("Normalizing business addresses...")
    df['norm_address'] = df['business_address'].apply(normalize_business_address)
    
    print("Extracting postal codes...")
    df['postal_code'] = df['business_address'].apply(extract_postal_code)
    
    print("Extracting generic numbers from address...")
    df['address_numbers'] = df['business_address'].apply(extract_numbers)
    
    return df

if __name__ == "__main__":
    # Test the normalizer on a small sample of the data to verify it works
    print("Testing Normalizer on a sample of Train Source 1...")
    sample_df = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=10)
    
    normalized_df = normalize_dataframe(sample_df)
    
    print("\nSample Results:")
    for _, row in normalized_df.iterrows():
        print(f"Original Name : {row['business_name']}")
        print(f"Norm Name     : {row['norm_name']}")
        print(f"Original Addr : {row['business_address']}")
        print(f"Norm Addr     : {row['norm_address']}")
        print(f"Postal Code   : {row['postal_code']}")
        print("-" * 50)
