import pandas as pd

df_data = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step_7_2_output.csv")
michelin = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/michlin.csv")

print(df_data.columns)
michelin.rename(columns={'poicode':'poi_code'},inplace=True)
michelin.dropna(subset=['poi_code'], inplace=True)
michelin['poi_code'] = michelin['poi_code'].apply(lambda x: x.split('_')[3])

EXCLUDED_DOMAINS = {
    'amazon.in','amazon.com','flipkart.com','myntra.com','snapdeal.com',
    'booking.com','agoda.com','expedia.com','makemytrip.com','airbnb.com',
    'zomato.com','swiggy.com','ubereats.com',
    'facebook.com','instagram.com','twitter.com','linkedin.com',
    'whatsapp.com','telegram.org','youtube.com',
    'paytm.com','phonepe.com','googlepay.com',
    'justdial.com','sulekha.com','indiamart.com',
    'bit.ly','tinyurl.com','goo.gl','t.co',
    'google.com','microsoft.com','apple.com'
}

EXCLUDED_DOMAIN_PATTERNS = [
    '.gov','.gov.','.nic','.nic.','.edu','.edu.','.ac','.ac.',
    'www.','mail.','admin.','login.','secure.',
    '.wordpress.','.wixsite.','.blogspot.'
]

EXCLUSIONS_TAKE_PRIORITY = True

def is_valid_domain(domain):

    if not domain or pd.isna(domain):
        return 0

    domain = str(domain).lower().strip()

    for excluded in EXCLUDED_DOMAINS:
        if excluded in domain:
            return 0

    for pattern in EXCLUDED_DOMAIN_PATTERNS:
        if pattern in domain:
            return 0

    return 1

df_data['has_domain'] = df_data['website_domain_name'].apply(is_valid_domain)
michelin_set = set(michelin['poi_code'])

# assign flag
df_data['michelin_star'] = df_data['poi_code'].astype(str).isin(michelin_set).astype(int)

df_data.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step8_data_output.csv", index=False)