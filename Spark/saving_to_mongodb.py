import requests
import math
from datetime import datetime
from pyspark.sql import SparkSession
# Εισαγωγή τύπων δεδομένων για τον καθορισμό της δομής του DataFrame
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, ArrayType, IntegerType
# Εισαγωγή συναρτήσεων SQL για την επεξεργασία στηλών και UDFs
from pyspark.sql.functions import col, udf, explode, lit
import sys
import os

# Ορίζουμε τη διαδρομή του Python interpreter για να διασφαλίσουμε ότι ο Spark 
# χρησιμοποιεί την ίδια έκδοση Python σε driver και workers
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- UNIFIED SCHEMA (Ενιαίο Σχήμα Δεδομένων) ---
# Καθορίζουμε τη δομή των δεδομένων μας ώστε να είναι κοινή για όλες τις πηγές (Coursera, OpenLibrary).
# Αυτό εξασφαλίζει τη συμβατότητα των δεδομένων πριν την εισαγωγή στη βάση.
UNIFIED_SCHEMA = StructType([
    StructField("title", StringType(), True),         # Ο τίτλος του μαθήματος ή του βιβλίου
    StructField("description", StringType(), True),   # Περιγραφή ή θέματα περιεχομένου
    StructField("category", StringType(), True),      # Η κατηγορία στην οποία ανήκει
    StructField("language", StringType(), True),      # Γλώσσα περιεχομένου
    StructField("level", StringType(), True),         # Επίπεδο δυσκολίας
    StructField("source_name", StringType(), True),   # Από ποια πλατφόρμα προήλθε (π.χ. Coursera)
    StructField("link", StringType(), True),          # URL για την πηγή
    StructField("last_updated", TimestampType(), True) # Χρονική σήμανση της συλλογής (ETL time)
])


# ==========================================
# 1. COURSERA LOGIC
# ==========================================

# Αυτή η συνάρτηση εκτελείται αυτόνομα σε κάθε worker του Spark.
# Δέχεται ένα offset (σημείο έναρξης) και φέρνει μια σελίδα 100 αποτελεσμάτων από το API.
def fetch_coursera_worker(start_offset):
    import requests
    from datetime import datetime
    results = []
    try:
        # Διεύθυνση του API της Coursera για τα μαθήματα
        url = "https://api.coursera.org/api/courses.v1"
        # Ορίζουμε τα πεδία που θέλουμε να μας επιστρέψει το API για να γεμίσουμε το Unified Schema
        params = {
            "fields": "name,slug,description,workload,primaryLanguages,partnerIds,domainTypes",
            "limit": 100,           # Μέγιστος αριθμός ανά κλήση
            "start": start_offset   # Από ποιο μάθημα να ξεκινήσει η λήψη
        }
        response = requests.get(url, params=params)
        if response.status_code != 200: return []
        
        data = response.json()
        # Επεξεργασία κάθε στοιχείου που επέστρεψε το API
        for item in data.get("elements", []):
            domains = item.get("domainTypes", [])
            # Αν υπάρχουν domain types, παίρνουμε το subdomainId ως κατηγορία, αλλιώς βάζουμε "General"
            category = domains[0].get("subdomainId", "General") if domains else "General"
            languages = item.get("primaryLanguages", ["en"])
            
            # Δημιουργία πλειάδας (tuple) που αντιστοιχεί ακριβώς στο UNIFIED_SCHEMA
            results.append((
                item.get("name", ""),
                item.get("description", ""),
                category,
                languages[0],
                "Unknown",
                "Coursera",
                f"https://www.coursera.org/learn/{item.get('slug', '')}",
                datetime.now()
            ))
    except Exception:
        return [] # Επιστροφή κενής λίστας σε περίπτωση σφάλματος για να μην καταρρεύσει η ροή
    return results


def fetch_all_coursera_df(spark):
    print("--- Starting Coursera Fetch ---")
    try:
        # Πρώτη κλήση στο API για να δούμε πόσα μαθήματα υπάρχουν συνολικά (paging metadata)
        meta = requests.get("https://api.coursera.org/api/courses.v1?limit=1").json()
        total = meta.get("paging", {}).get("total", 0)
        print(f"Coursera total courses: {total}")
        if total == 0: return spark.createDataFrame([], UNIFIED_SCHEMA)

        # Δημιουργούμε ένα Spark DataFrame που περιέχει μόνο μια στήλη με τα offsets (0, 100, 200...)
        # Αυτό επιτρέπει στο Spark να μοιράσει τις κλήσεις API σε πολλούς workers ταυτόχρονα.
        offsets_df = spark.range(0, total, 100).withColumnRenamed("id", "offset")
        
        # Ανακατανέμουμε τα δεδομένα (repartition) για να αυξήσουμε τον παραλληλισμό
        offsets_df = offsets_df.repartition((total // 100) // 2 + 1)

        # Μετατρέπουμε τη συνάρτηση fetch_coursera_worker σε Spark UDF (User Defined Function)
        coursera_udf = udf(fetch_coursera_worker, ArrayType(UNIFIED_SCHEMA))

        # Εφαρμόζουμε την UDF: Για κάθε offset, καλείται το API και επιστρέφεται μια λίστα μαθημάτων.
        # Με το explode(), κάθε στοιχείο της λίστας γίνεται ξεχωριστή γραμμή στο DataFrame.
        return offsets_df \
            .withColumn("batch_data", coursera_udf(col("offset"))) \
            .select(explode(col("batch_data")).alias("course")) \
            .select("course.*")
    except Exception as e:
        print(f"Error in Coursera: {e}")
        return spark.createDataFrame([], UNIFIED_SCHEMA)


# ==========================================
# 2. OPEN LIBRARY LOGIC
# ==========================================

# Λίστα με τα "θέματα" (subjects) που θα αναζητήσουμε στο Open Library για να γεμίσουμε τη βάση μας
CATEGORIES = [
    "computer_science", "mathematics", "physics", "chemistry", "biology",
    "medicine", "engineering", "architecture", "education", "management",
    "history", "art", "music", "law", "economics",
    "psychology", "philosophy", "political_science", "anthropology", "sociology"
]

def fetch_openlib_worker_multi(category, offset):
    import requests
    from datetime import datetime

    results = []
    try:
        # Κατασκευή URL δυναμικά με βάση την κατηγορία που επεξεργάζεται ο worker
        url = f"https://openlibrary.org/subjects/{category}.json"
        params = {"limit": 100, "offset": offset, "details": "true"}

        # Κλήση API με timeout για αποφυγή μεγάλων αναμονών αν η πηγή αργεί
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return []

        data_json = response.json()
        # Καθαρίζουμε το slug (π.χ. computer_science -> Computer Science) για ομορφότερη εμφάνιση
        clean_category = category.replace("_", " ").title()

        # Μετασχηματίζουμε τα βιβλία σε μορφή συμβατή με το σχήμα των μαθημάτων μας
        for item in data_json.get("works", []):
            title = item.get("title", "Unknown Title")
            subjects = item.get("subject", [])
            # Χρησιμοποιούμε τα πρώτα 5 subjects ως "περιγραφή" αφού τα βιβλία δεν έχουν πάντα έτοιμο description
            description = f"Topics: {', '.join(subjects[:5])}" if subjects else "No description available"
            key = item.get("key", "")
            link = f"https://openlibrary.org{key}" if key else "https://openlibrary.org"

            results.append((
                title,
                description,
                clean_category,
                "en",
                "N/A",
                "OpenLibrary",
                link,
                datetime.now()
            ))
    except Exception:
        return []

    return results


def fetch_all_openlib_multi_df(spark):
    print("--- Starting Multi-Category Open Library Fetch ---")

    # 1. Δημιουργούμε DataFrame με τις κατηγορίες που ορίσαμε στη λίστα
    cats_df = spark.createDataFrame([(c,) for c in CATEGORIES], ["category_slug"])

    # 2. Δημιουργούμε DataFrame για τα offsets (θα ζητήσουμε μέχρι 1000 βιβλία ανά κατηγορία)
    offsets_df = spark.range(0, 1000, 100).withColumnRenamed("id", "offset_val")

    # 3. Cross Join: Συνδυάζουμε κάθε κατηγορία με κάθε offset.
    # Π.χ. (computer_science, 0), (computer_science, 100), ..., (mathematics, 0), κλπ.
    # Αυτό δημιουργεί συνολικά 200 ανεξάρτητα tasks (20 κατηγορίες * 10 σελίδες).
    tasks_df = cats_df.crossJoin(offsets_df)

    # Μοιράζουμε τα 200 tasks σε 40 partitions για να εκτελούνται παράλληλα
    tasks_df = tasks_df.repartition(40)

    # Ορισμός της UDF που δέχεται δύο παραμέτρους (όνομα κατηγορίας και offset)
    openlib_udf = udf(fetch_openlib_worker_multi, ArrayType(UNIFIED_SCHEMA))

    # 4. Εκτέλεση του Harvesting
    df_exploded = tasks_df \
        .withColumn("batch_data", openlib_udf(col("category_slug"), col("offset_val"))) \
        .select(explode(col("batch_data")).alias("book")) \
        .select("book.*")

    return df_exploded


# ==========================================
# 3. MAIN EXECUTION (Κεντρική Εκτέλεση)
# ==========================================

# Δημιουργία του Spark Session. 
# Εδώ προσθέτουμε το απαραίτητο πακέτο για τη σύνδεση με τη MongoDB.
spark = SparkSession.builder \
    .appName("Saving Courses Multi-Category") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/coursesApplication.courses") \
    .getOrCreate()

# Βήμα 1: Συλλογή δεδομένων από την Coursera (Spark Transformation)
df_coursera = fetch_all_coursera_df(spark)

# Βήμα 2: Συλλογή δεδομένων από την OpenLibrary (Spark Transformation)
df_openlib = fetch_all_openlib_multi_df(spark)

# Βήμα 3: Ένωση (Union) των δύο DataFrames. 
# Επειδή έχουν το ίδιο σχήμα (UNIFIED_SCHEMA), η ένωση γίνεται άμεσα.
print("Unioning DataFrames...")
final_df = df_coursera.union(df_openlib)

# Βήμα 4: Αποθήκευση στη MongoDB. 
# Χρησιμοποιούμε το mode("append") για να προσθέσουμε τα νέα δεδομένα χωρίς να διαγράψουμε τα παλιά.
print("Writing to MongoDB...")
try:
    final_df.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", "coursesApplication") \
        .option("collection", "courses") \
        .save()
    print("Data written successfully!")
except Exception as e:
    print(f"Error writing to MongoDB: {e}")

# Τερματισμός του Spark Session για απελευθέρωση πόρων
spark.stop()
