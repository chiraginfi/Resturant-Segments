#!/usr/bin/env python3
"""
Extract bar menu items for various alcohol brands including Jack Daniels, Woodford, 
Herradura, Black Label, Jameson, Red Label, and many others from Brown-Forman 
Gemini production JSON results.

Reads all JSON files in the given directory (filename = restaurant identifier),
filters items whose name contains any of the target brands, and writes
a CSV with columns: name (restaurant), category, items_name, price, brand_name.

Categories include specific cocktail types (Martini, Old Fashioned, Mojito, Picante)
and general alcohol categories (gin, whisky, rum, vodka, etc.).
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


# (pattern, display name); "jack daniel" avoids Jack Frost, Jacky Marteau, etc.
BRANDS = [
    (re.compile(r"jack\s+daniel", re.I), "Jack Daniels"),
    (re.compile(r"woodford", re.I), "Woodford"),
    (re.compile(r"herradura", re.I), "Herradura"),
    (re.compile(r"glendronach", re.I), "Glendronach"),
    (re.compile(r"black\s+label", re.I), "Black Label"),
    (re.compile(r"johnnie\s*walker.*blue\s*label|blue\s*label", re.I), "Blue Label"),
    (re.compile(r"jameson", re.I), "Jameson"),
    (re.compile(r"red\s+label", re.I), "Red Label"),

    (re.compile(r"abk6", re.I), "ABK6"),
    (re.compile(r"absente", re.I), "Absente"),
    (re.compile(r"acqua\s+bianca", re.I), "Acqua Bianca"),
    (re.compile(r"adriatico", re.I), "Adriatico"),
    (re.compile(r"aelred", re.I), "Aelred"),
    (re.compile(r"agwa", re.I), "Agwa"),
    (re.compile(r"akashi[\-\s]?tai", re.I), "Akashi-Tai"),
    (re.compile(r"algebra", re.I), "Algebra"),
    (re.compile(r"alma\s+finca", re.I), "Alma Finca"),
    (re.compile(r"amargo", re.I), "Amargo"),
    (re.compile(r"amarguinha", re.I), "Amarguinha"),
    (re.compile(r"amendoa\s+amarga\s+amerlinha", re.I), "Amendoa Amarga Amerlinha"),
    (re.compile(r"ancho\s+reyes", re.I), "Ancho Reyes"),
    (re.compile(r"antica", re.I), "Antica"),
    (re.compile(r"archers", re.I), "Archers"),
    (re.compile(r"arhumatic", re.I), "Arhumatic"),
    (re.compile(r"arran", re.I), "Arran"),
    (re.compile(r"asbach", re.I), "Asbach"),
    (re.compile(r"audemus", re.I), "Audemus"),
    (re.compile(r"aurum", re.I), "Aurum"),
    (re.compile(r"axia", re.I), "Axia"),
    (re.compile(r"baileys", re.I), "Baileys"),
    (re.compile(r"balcones", re.I), "Balcones"),
    (re.compile(r"bardinet", re.I), "Bardinet"),
    (re.compile(r"b[äa]renj[aä]ger", re.I), "Bärenjäger"),
    (re.compile(r"batida\s+de\s+coco", re.I), "Batida de Coco"),
    (re.compile(r"becherovka", re.I), "Becherovka"),
    (re.compile(r"beirao", re.I), "Beirao"),
    (re.compile(r"benedictine", re.I), "Benedictine"),
    (re.compile(r"berentzen", re.I), "Berentzen"),
    (re.compile(r"bernal", re.I), "Bernal"),
    (re.compile(r"bernard", re.I), "Bernard"),
    (re.compile(r"bigallet", re.I), "Bigallet"),
    (re.compile(r"bitter\s+truth", re.I), "Bitter Truth"),
    (re.compile(r"bittermens", re.I), "Bittermens"),
    (re.compile(r"boatyard", re.I), "Boatyard"),
    (re.compile(r"bols", re.I), "Bols"),
    (re.compile(r"borghetti", re.I), "Borghetti"),
    (re.compile(r"born\s+irish", re.I), "Born Irish"),
    (re.compile(r"bourg", re.I), "Bourg"),
    (re.compile(r"bouvery", re.I), "Bouvery"),
    (re.compile(r"briottet", re.I), "Briottet"),
    (re.compile(r"broken\s+bones", re.I), "Broken Bones"),
    (re.compile(r"bruadar", re.I), "Bruadar"),
    (re.compile(r"cacao\s+blanc", re.I), "Cacao Blanc"),
    (re.compile(r"cacique", re.I), "Cacique"),
    (re.compile(r"caffe\s+borghetti", re.I), "Caffe Borghetti"),
    (re.compile(r"camps", re.I), "Camps"),
    (re.compile(r"canouan", re.I), "Canouan"),
    (re.compile(r"canton", re.I), "Canton"),
    (re.compile(r"cappelletti", re.I), "Cappelletti"),
    (re.compile(r"capra", re.I), "Capra"),
    (re.compile(r"caravella", re.I), "Caravella"),
    (re.compile(r"cardamaro", re.I), "Cardamaro"),
    (re.compile(r"cariel", re.I), "Cariel"),
    (re.compile(r"carmencita", re.I), "Carmencita"),
    (re.compile(r"chambord", re.I), "Chambord"),
    (re.compile(r"chartreuse\s+green", re.I), "Chartreuse Green"),
    (re.compile(r"chartreuse\s+yellow", re.I), "Chartreuse Yellow"),
    (re.compile(r"cinzano", re.I), "Cinzano"),
    (re.compile(r"cointreau", re.I), "Cointreau"),
    (re.compile(r"contratto", re.I), "Contratto"),
    (re.compile(r"coruba", re.I), "Coruba"),
    (re.compile(r"cynar", re.I), "Cynar"),
    (re.compile(r"dekuyper", re.I), "DeKuyper"),
    (re.compile(r"disaronno", re.I), "Disaronno"),
    (re.compile(r"drambuie", re.I), "Drambuie"),
    (re.compile(r"el\s+dorado", re.I), "El Dorado"),
    (re.compile(r"fernet[\-\s]?branca", re.I), "Fernet-Branca"),
    (re.compile(r"fiorente", re.I), "Fiorente"),
    (re.compile(r"frangelico", re.I), "Frangelico"),
    (re.compile(r"galliano", re.I), "Galliano"),
    (re.compile(r"g[ée]n[ée]py\s+des\s+alpes", re.I), "Génépy des Alpes"),
    (re.compile(r"goldwasser", re.I), "Goldwasser"),
    (re.compile(r"grand\s+marnier", re.I), "Grand Marnier"),
    (re.compile(r"griottine", re.I), "Griottine"),
    (re.compile(r"hpnotiq", re.I), "Hpnotiq"),
    (re.compile(r"irish\s+mist", re.I), "Irish Mist"),
    (re.compile(r"j[aä]germeister", re.I), "Jagermeister"),
    (re.compile(r"kahlua", re.I), "Kahlua"),
    (re.compile(r"kirschwasser", re.I), "Kirschwasser"),
    (re.compile(r"licor\s+43", re.I), "Licor 43"),
    (re.compile(r"luxardo\s+bitter", re.I), "Luxardo Bitter"),
    (re.compile(r"mandarine\s+napoleon", re.I), "Mandarine Napoleon"),
    (re.compile(r"maraschino", re.I), "Maraschino"),
    (re.compile(r"meletti", re.I), "Meletti"),
    (re.compile(r"midori", re.I), "Midori"),
    (re.compile(r"minttu", re.I), "Minttu"),
    (re.compile(r"nocello", re.I), "Nocello"),
    (re.compile(r"passoa", re.I), "Passoa"),
    (re.compile(r"pisang\s+ambon", re.I), "Pisang Ambon"),
    (re.compile(r"ponche\s+caballero", re.I), "Ponche Caballero"),
    (re.compile(r"ramazzotti", re.I), "Ramazzotti"),
    (re.compile(r"rumchata", re.I), "RumChata"),
    (re.compile(r"sambuca", re.I), "Sambuca"),
    (re.compile(r"st[\.\s]?germain", re.I), "St Germain"),
    (re.compile(r"stag'?s\s+breath", re.I), "Stag's Breath"),
    (re.compile(r"t?ia\s+maria", re.I), "Tia Maria"),
    (re.compile(r"umeshu", re.I), "Umeshu"),
    (re.compile(r"ypioca", re.I), "Ypioca"),
    (re.compile(r"zubrowka", re.I), "Zubrowka"),
    (re.compile(r"glenfiddich", re.I), "Glenfiddich"),
    (re.compile(r"glenlivet", re.I), "The Glenlivet"),
    (re.compile(r"singleton", re.I), "The Singleton"),
    (re.compile(r"wood\s*burns?", re.I), "Wood Burns"),
    (re.compile(r"iconiq", re.I), "Iconiq"),
    (re.compile(r"legacy", re.I), "Legacy"),
    (re.compile(r"wolf\s*stone", re.I), "Wolf Stone"),

    # --- Indian & India-popular brands ---

    # Indian Whisky
    (re.compile(r"royal\s+stag", re.I), "Royal Stag"),
    (re.compile(r"imperial\s+blue", re.I), "Imperial Blue"),
    (re.compile(r"officer['s]*\s+choice", re.I), "Officer's Choice"),
    (re.compile(r"mcdowell['s]*\s+no[\.\s]?1", re.I), "McDowell's No.1"),
    (re.compile(r"mcdowell", re.I), "McDowell's"),
    (re.compile(r"blenders\s+pride", re.I), "Blenders Pride"),
    (re.compile(r"royal\s+challenge", re.I), "Royal Challenge"),
    (re.compile(r"antiquity\s+blue", re.I), "Antiquity Blue"),
    (re.compile(r"antiquity\s+rare", re.I), "Antiquity Rare"),
    (re.compile(r"antiquity", re.I), "Antiquity"),
    (re.compile(r"8\s*pm\s+whisky", re.I), "8 PM Whisky"),
    (re.compile(r"signature\s+rare", re.I), "Signature Rare"),
    (re.compile(r"hayward['s]*\s+fine", re.I), "Hayward's Fine"),
    (re.compile(r"director['s]*\s+special", re.I), "Director's Special"),
    (re.compile(r"black\s+dog", re.I), "Black Dog"),
    (re.compile(r"black\s+&\s+white", re.I), "Black & White"),
    (re.compile(r"100\s+pipers", re.I), "100 Pipers"),
    (re.compile(r"vat\s+69", re.I), "VAT 69"),
    (re.compile(r"something\s+special", re.I), "Something Special"),
    (re.compile(r"chivas\s+regal", re.I), "Chivas Regal"),
    (re.compile(r"teachers", re.I), "Teacher's"),
    (re.compile(r"ballantine", re.I), "Ballantine's"),
    (re.compile(r"the\s+singleton", re.I), "The Singleton"),
    (re.compile(r"paul\s+john", re.I), "Paul John"),
    (re.compile(r"amrut", re.I), "Amrut"),
    (re.compile(r"rampur", re.I), "Rampur"),
    (re.compile(r"indri", re.I), "Indri"),
    (re.compile(r"greater\s+than", re.I), "Greater Than"),
    (re.compile(r"godawan", re.I), "Godawan"),

    # Indian Rum
    (re.compile(r"old\s+monk", re.I), "Old Monk"),
    (re.compile(r"contessa\s+rum", re.I), "Contessa Rum"),
    (re.compile(r"mcdowell['s]*\s+rum", re.I), "McDowell's Rum"),
    (re.compile(r"bacardi", re.I), "Bacardi"),
    (re.compile(r"captain\s+morgan", re.I), "Captain Morgan"),
    (re.compile(r"camikara", re.I), "Camikara"),
    (re.compile(r"maka\s+zai", re.I), "Maka Zai"),
    (re.compile(r"segredo", re.I), "Segredo Aldeia"),
    (re.compile(r"short\s+story", re.I), "Short Story"),
    (re.compile(r"hercules", re.I), "Hercules"),

    # Indian Brandy
    (re.compile(r"honey\s+bee", re.I), "Honey Bee"),
    (re.compile(r"morpheus", re.I), "Morpheus"),
    (re.compile(r"mansion\s+house", re.I), "Mansion House"),
    (re.compile(r"old\s+admiral", re.I), "Old Admiral"),

    # Indian Gin
    (re.compile(r"hapusa", re.I), "Hapusa"),
    (re.compile(r"stranger\s+&\s+sons", re.I), "Stranger & Sons"),
    (re.compile(r"jaisalmer", re.I), "Jaisalmer"),
    (re.compile(r"rokk\s+gin", re.I), "Rokk Gin"),
    (re.compile(r"nao\s+spirits", re.I), "Nao Spirits"),
    (re.compile(r"tickle\s+gin", re.I), "Tickle Gin"),
    (re.compile(r"hendrick'?s", re.I), "Hendrick’s"),
    (re.compile(r"roku", re.I), "Roku"),
    (re.compile(r"tanqueray", re.I), "Tanqueray"),
    (re.compile(r"bombay\s+sapphire", re.I), "Bombay Sapphire"),
    (re.compile(r"samsara", re.I), "Samsara"),
    (re.compile(r"terai", re.I), "Terai"),
    (re.compile(r"beefeater", re.I), "Beefeater"),
    (re.compile(r"gordon'?s", re.I), "Gordon’s"),
    (re.compile(r"blue\s+riband", re.I), "Blue Riband"),

    # Indian Beer
    (re.compile(r"kingfisher", re.I), "Kingfisher"),
    (re.compile(r"tuborg", re.I), "Tuborg"),
    (re.compile(r"carlsberg", re.I), "Carlsberg"),
    (re.compile(r"heineken", re.I), "Heineken"),
    (re.compile(r"corona", re.I), "Corona"),
    (re.compile(r"budweiser|bud\s*magnum", re.I), "Budweiser"),
    (re.compile(r"foster['s]*", re.I), "Foster's"),
    (re.compile(r"hayward['s]*\s+5000", re.I), "Hayward's 5000"),
    (re.compile(r"royal\s+challenge\s+beer", re.I), "Royal Challenge Beer"),
    (re.compile(r"bira\s+91", re.I), "Bira 91"),
    (re.compile(r"white\s+owl", re.I), "White Owl"),
    (re.compile(r"simba", re.I), "Simba"),
    (re.compile(r"godfather", re.I), "Godfather"),
    (re.compile(r"hoegaarden", re.I), "Hoegaarden"),
    (re.compile(r"stella\s*artois", re.I), "Stella Artois"),
    (re.compile(r"erdinger", re.I), "Erdinger"),
    (re.compile(r"medusa", re.I), "Medusa"),
    (re.compile(r"beeyoung", re.I), "BeeYoung"),

    # Indian Wine
    (re.compile(r"sula", re.I), "Sula"),
    (re.compile(r"grover\s+zampa", re.I), "Grover Zampa"),
    (re.compile(r"fratelli", re.I), "Fratelli"),
    (re.compile(r"york\s+winery", re.I), "York Winery"),
    (re.compile(r"soma\s+vine", re.I), "Soma Vine"),

    # Indian Vodka & Other
    (re.compile(r"magic\s+moments", re.I), "Magic Moments"),
    (re.compile(r"romanov", re.I), "Romanov"),
    (re.compile(r"white\s+mischief", re.I), "White Mischief"),
    (re.compile(r"smirnoff", re.I), "Smirnoff"),
    (re.compile(r"absolut", re.I), "Absolut"),
    (re.compile(r"grey\s+goose", re.I), "Grey Goose"),
    (re.compile(r"ketel\s+one", re.I), "Ketel One"),
    (re.compile(r"ciroc", re.I), "Ciroc"),
    (re.compile(r"belvedere", re.I), "Belvedere"),
    (re.compile(r"skyy", re.I), "Skyy"),
    (re.compile(r"smoke", re.I), "Smoke"),

    #feni
    (re.compile(r"goenchi", re.I), "Goenchi"),
    (re.compile(r"cazulo", re.I), "Cazulo"),
    (re.compile(r"moji", re.I), "Moji"),
    (re.compile(r"volando", re.I), "Volando"),
    (re.compile(r"aani\s+ek", re.I), "Aani Ek"),
    (re.compile(r"fidalgo", re.I), "Fidalgo"),
    (re.compile(r"big\s+boss", re.I), "Big Boss"),
    (re.compile(r"tinto", re.I), "Tinto"),
    (re.compile(r"cazcar", re.I), "Cazcar"),
    (re.compile(r"patrao", re.I), "Patrao"),

    # Existing generic rules (keep at end)
    (re.compile(r"cashew\s+feni", re.I), "Cashew Feni"),
    (re.compile(r"coconut\s+feni", re.I), "Coconut Feni"),
    (re.compile(r"feni", re.I), "Feni"),

    #brandy
    (re.compile(r"remy\s+martin", re.I), "Remy Martin"),
    (re.compile(r"hennessy", re.I), "Hennessy"),
    (re.compile(r"martell", re.I), "Martell"),
    (re.compile(r"courvoisier", re.I), "Courvoisier"),
    (re.compile(r"st[\-\s]?remy", re.I), "St-Rémy"),
    (re.compile(r"roulette", re.I), "Roulette"),
    (re.compile(r"bejois", re.I), "Bejois"),
    (re.compile(r"constantino", re.I), "Constantino"),
]


def get_brand_name(item_name: str) -> str | None:
    """Return the first matching brand display name, or None."""
    if not item_name:
        return None
    for pattern, brand_name in BRANDS:
        if pattern.search(item_name):
            return brand_name
    return None


# Keywords to standardize categories based on original category field
CATEGORY_KEYWORDS = {
    "gin": [r"\bgin\b", r"london\s+dry", r"plymouth"],
    "whisky": [r"\bwhisk[ey]y\b", r"\bscotch\b", r"\bbourbon\b", r"\brye\b", r"\bsingle\s+malt\b", r"\bblended\b", 
               r"\bisle\b", r"\bspeyside\b", r"\bhighland\b", r"\bislay\b"],
    "rum": [r"\brum\b", r"\brhum\b", r"\bcachaça\b", r"\bspiced\b", r"\bdark\b", r"\bwhite\b",
            r"\baged\b", r"\bpuerto\s+ric", r"\bjamaican\b", r"\bbarbados"],
    "vodka": [r"\bvodka\b"],
    "tequila": [r"\btequila\b", r"\bblanco\b", r"\breposado\b", r"\bañejo\b", r"\bsilver\b", r"\bgold\b",
                r"\b100%\s+agave\b", r"\bajave\b"],
    "brandy": [r"\bbrandy\b", r"\bcognac\b", r"\barmagnac\b", r"\bpisco\b", 
               r"\bvsop\b", r"\bxo\b", r"\bhennessy\b", r"\bmartell\b"],
    "soju": [r"\bsoju\b", r"\bkorean\b.*spirit"],
    "absinthe": [r"\babsinthe\b", r"\babsenta\b", r"\bwormwood\b"],
    "mezcal": [r"\bmezcal\b", r"\bmezecal\b", r"\boaxaca\b"],
    "martini": [r"\bmartini\b"],
    "old_fashioned": [r"\bold\s+fashioned\b"],
    "mojito": [r"\bmojito\b"],
    "picante": [r"\bpicante\b"],
    "other_cocktails": [r"\bcocktail\b", r"\bcocktails\b", r"\bmocktail\b", r"\bmocktails\b"],
    "beer": [r"\bbeer\b", r"\bbeers\b"],
    "wine": [r"\bwine\b", r"\bwines\b"],
    "spirits": [r"\bspirit\b", r"\bspirits\b"],
}

BRAND_REGEX_CATEGORY_MAP = [
    # feni
    (re.compile(
        r"goenchi|cazulo|moji|volando|aani\s*ek|fidalgo|big\s*boss|tinto|cazcar|patrao|cashew\s*feni|coconut\s*feni|\bfeni\b",
        re.I
    ), "feni"),
    # whisky
    (re.compile(r"amrut|paul\s+john|indri|rampur|godawan|chivas|singleton|glenfiddich|glenlivet|jameson|jack\s+daniel|black\s+dog|100\s+pipers|teacher|ballantine|wood\s+burns|antiquity|signature|blenders\s+pride|royal\s+challenge|royal\s+stag|imperial\s+blue|mcdowell|officer|8\s*pm|director|vat\s+69|black\s*&\s*white|something\s+special|iconiq|legacy|wolf\s+stone", re.I), "whisky"),

    # gin
    (re.compile(r"hapusa|jaisalmer|stranger\s*&\s*sons|greater\s+than|hendrick|roku|tanqueray|bombay\s+sapphire|samsara|terai|beefeater|gordon|blue\s+riband", re.I), "gin"),

    # vodka
    (re.compile(r"grey\s+goose|belvedere|ciroc|smoke|absolut|ketel\s+one|skyy|smirnoff|magic\s+moments|romanov|white\s+mischief", re.I), "vodka"),

    # brandy
    (re.compile(r"remy\s+martin|hennessy|martell|courvoisier|st[\-\s]?remy|morpheus|roulette|mansion\s+house|honey\s+bee|old\s+admiral|bejois|constantino", re.I), "brandy"),

    # beer
    (re.compile(r"corona|hoegaarden|heineken|bira\s*91|simba|budweiser|carlsberg|kingfisher|stella|erdinger|medusa|tuborg|haywards\s*5000|godfather|beeyoung", re.I), "beer"),

    # rum
    (re.compile(r"camikara|maka\s+zai|segredo|short\s+story|old\s+monk|bacardi|captain\s+morgan|hercules|contessa", re.I), "rum"),
]

def extract_avg_price(item: dict):
    prices = []

    for key, value in item.items():
        if key.startswith("price") and value is not None:
            try:
                prices.append(float(value))
            except:
                pass

    if not prices:
        return ""

    return sum(prices) / len(prices)


def get_cocktail_category_from_item_name(item_name: str) -> str | None:
    """
    Check if item name contains specific cocktail types.
    Returns martini, old_fashioned, mojito, picante, or None.
    """
    if not item_name:
        return None
    
    item_lower = item_name.lower()
    
    # Check for specific cocktail types in item name
    cocktail_patterns = {
        "martini": [r"\bmartini\b"],
        "old_fashioned": [r"\bold\s+fashioned\b"],
        "mojito": [r"\bmojito\b"],
        "picante": [r"\bpicante\b"],
    }
    
    for cocktail_type, patterns in cocktail_patterns.items():
        for pattern in patterns:
            if re.search(pattern, item_lower, re.I):
                return cocktail_type
    
    return None


def get_standardized_category(original_category: str, item_name: str = "") -> str:
    """
    Enhanced category detection:
    Priority:
    1. Cocktail name
    2. Brand regex (NEW)
    3. Original category keywords
    """

    # --- 1. Cocktail detection ---
    cocktail_from_name = get_cocktail_category_from_item_name(item_name)
    if cocktail_from_name:
        return cocktail_from_name

    # --- 2. BRAND-BASED CATEGORY (NEW LOGIC) ---
    if item_name:
        for pattern, category in BRAND_REGEX_CATEGORY_MAP:
            if pattern.search(item_name):
                return category

    # --- 3. ORIGINAL CATEGORY ---
    if original_category:
        category_lower = original_category.lower()

        for standard_category, patterns in CATEGORY_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, category_lower, re.I):
                    return standard_category

    return "other"


def _has_numeric_price(price) -> bool:
    if price is None or price == "":
        return False
    try:
        float(price) if not isinstance(price, (int, float)) else price
        return True
    except (ValueError, TypeError):
        return False


def restaurant_name_from_path(path: Path) -> str:
    """Derive restaurant name from filename (e.g. belugamumbai.json -> belugamumbai)."""
    return path.stem


def extract_brand_items_from_file(json_path: Path) -> tuple[list[dict], dict]:
    """
    Load one JSON file and return (rows for CSV, stats_dict).
    stats_dict: alcohol_items, alcohol_with_brand, non_alcohol_items, alcohol_with_price, non_alcohol_with_price
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    bar_categories = data.get("bar_categories") or []
    has_bar_categories = len(bar_categories) > 0

    restaurant = data.get("restaurant") or restaurant_name_from_path(json_path)
    rows = []
    filename = json_path.name

    stats = {
        "has_bar_categories": has_bar_categories,
        "alcohol_items": 0,
        "alcohol_with_brand": 0,
        "alcohol_with_price": 0,
        "non_alcohol_items": 0,
        "non_alcohol_with_price": 0,
    }

    for cat in bar_categories:
        original_category = cat.get("category") or ""

        for item in cat.get("items") or []:
            item_name = item.get("name") or ""
            # Check category using both original category and item name
            standardized_category = get_standardized_category(original_category, item_name)
            # Alcohol = standardized category is not "other"; other = non-alcohol
            is_non_alc = standardized_category == "other"
            
            brand_name = get_brand_name(item_name)
            if brand_name is None:
                brand_name = "other"
            has_brand = brand_name != "other"
            price = extract_avg_price(item)
            if price is None:
                price = ""
            has_price = _has_numeric_price(price)

            if is_non_alc:
                stats["non_alcohol_items"] += 1
                if has_price:
                    stats["non_alcohol_with_price"] += 1
            else:
                stats["alcohol_items"] += 1
                if has_brand:
                    stats["alcohol_with_brand"] += 1
                if has_price:
                    stats["alcohol_with_price"] += 1

            # If category is "other", only include items with a known brand (exclude brand_name "other")
            if standardized_category == "other" and brand_name == "other":
                continue
            rows.append({
                "name": restaurant,
                "category": standardized_category,
                "items_name": item_name,
                "price": price,
                "brand_name": brand_name,
                "filename": filename,
            })

    return rows, stats


def main() -> None:
    results_dir = Path("/mnt/data/image_recognition/restaurant_menu/brownforman_gemini_production_results")
    out_csv = results_dir / "brownforman_brand_items.csv"
    out_csv_grouped = results_dir / "brownforman_brand_items_by_restaurant_category.csv"
    out_csv_grouped_brand = results_dir / "brownforman_brand_items_by_brand_category.csv"
    out_csv_menu_stats = results_dir / "brownforman_menu_stats.csv"

    all_rows = []
    menu_stats_list = []
    menus_crawled = 0
    menus_with_bar_categories = 0
    agg = {
        "alcohol_items": 0,
        "alcohol_with_brand": 0,
        "alcohol_with_price": 0,
        "non_alcohol_items": 0,
        "non_alcohol_with_price": 0,
    }

    for json_path in sorted(results_dir.glob("*.json")):
        if json_path.name.startswith("resume_"):
            continue
        menus_crawled += 1
        try:
            rows, stats = extract_brand_items_from_file(json_path)
            all_rows.extend(rows)
            if stats.get("has_bar_categories"):
                menus_with_bar_categories += 1
            for k in agg:
                if k != "has_bar_categories" and k in stats:
                    agg[k] += stats[k]
            menu_stats_list.append({
                "menu": json_path.name,
                "alcohol_items": stats.get("alcohol_items", 0),
                "alcohol_with_brand": stats.get("alcohol_with_brand", 0),
                "non_alcohol_items": stats.get("non_alcohol_items", 0),
            })
        except Exception as e:
            print(f"Warning: skip {json_path.name}: {e}")

    print("\n--- Menu / item counts ---")
    print(f"Number of menus crawled: {menus_crawled}")
    print(f"Number of menus with non-empty bar_categories: {menus_with_bar_categories}")
    print(f"Number of alcohol items (in bar_categories): {agg['alcohol_items']}")
    print(f"Number of alcohol items with brand identified: {agg['alcohol_with_brand']}")
    print(f"Number of non-alcohol items: {agg['non_alcohol_items']}")
    print(f"Number of alcohol items with price: {agg['alcohol_with_price']}")
    print(f"Number of non-alcohol items with price: {agg['non_alcohol_with_price']}")
    print("---\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "items_name", "price", "brand_name", "filename"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_csv}")

    # Per-menu stats: alcohol_items, alcohol_with_brand, non_alcohol_items per JSON
    with open(out_csv_menu_stats, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["menu", "alcohol_items", "alcohol_with_brand", "non_alcohol_items"],
        )
        writer.writeheader()
        writer.writerows(menu_stats_list)
    print(f"Wrote {len(menu_stats_list)} menu stats to {out_csv_menu_stats}")

    def min_max_from_prices(prices: list) -> tuple:
        numeric_prices = []
        for p in prices:
            if p == "" or p is None:
                continue
            try:
                numeric_prices.append(float(p) if not isinstance(p, (int, float)) else p)
            except (ValueError, TypeError):
                pass
        if not numeric_prices:
            return ("", "")
        return (min(numeric_prices), max(numeric_prices))

    # Group by name, category -> min_price, max_price
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in all_rows:
        key = (row["name"], row["category"])
        grouped[key].append(row["price"])

    group_rows = []
    for (name, category), prices in sorted(grouped.items()):
        min_price, max_price = min_max_from_prices(prices)
        group_rows.append({
            "name": name,
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
        })

    with open(out_csv_grouped, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "min_price", "max_price"])
        writer.writeheader()
        writer.writerows(group_rows)

    print(f"Wrote {len(group_rows)} grouped rows to {out_csv_grouped}")

    # Group by brand_name, category -> min_price, max_price
    grouped_brand: dict[tuple[str, str], list] = defaultdict(list)
    for row in all_rows:
        key = (row["brand_name"], row["category"])
        grouped_brand[key].append(row["price"])

    group_brand_rows = []
    for (brand_name, category), prices in sorted(grouped_brand.items()):
        min_price, max_price = min_max_from_prices(prices)
        group_brand_rows.append({
            "brand_name": brand_name,
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
        })

    with open(out_csv_grouped_brand, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["brand_name", "category", "min_price", "max_price"])
        writer.writeheader()
        writer.writerows(group_brand_rows)

    print(f"Wrote {len(group_brand_rows)} grouped rows (by brand_name, category) to {out_csv_grouped_brand}")


if __name__ == "__main__":
    main()
