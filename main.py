import os
from flask import Flask
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from dateutil import parser
import db_connection
import psycopg2
import spacy
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
import numpy as np
import torch
app = Flask(__name__)
CORS(app)


nlp = spacy.load("en_core_web_sm")


model = SentenceTransformer('all-MiniLM-L6-v2')

data = {
    "sales_performer": [
        "get top sales performer",
        " monthly top sales performer",
        "query sales guy ",
        "retrieve top salesperson",
        "monthly leading sales performance",
        "Who achieved the highest sales this month?",
        "query sales leader"

        "identify monthly least performer",
        "query sales underperformer",
        "retrieve worst salesperson",
        "monthly sales underperformance",
        "Who achieved the lowest sales this month?",


    ],
    
    "total_revenue": [
        "get monthly total revenue",
        "query monthly revenue",
        "retrieve total revenue",
        "calculate monthly revenue",
        "monthly revenue summary",
        "What’s the total sales revenue for this month?",
        

    ],
    "sold_product": [
        "get most sold product",
        "identify top selling product",
        "query monthly best seller",
        "retrieve most popular product",
        "monthly sales  product",
        "Which product sold the most this month?",

        "identify not a top selling product",
        "query monthly worst seller",
        "retrieve monthly least product",
        "worst sold product",
        "Which product sold the least this month?",

    ],
    
    "revenue_client": [
        "get top revenue client",
        "identify highest revenue client",
        "query monthly top client",
        "retrieve client with highest revenue",
        "monthly client revenue leader",
        "Which client brought in the most revenue this month?",

"get least revenue client",
        "identify lowest revenue client",
        "query monthly least client",
        "retrieve client with lowest revenue",
        "monthly client revenue leader with low revenue",
        "Which client brought in the least revenue this month?",


    ],
   
    
    "sales_comparison": [
        "compare monthly sales",
        "sales comparison monthly",
        "retrieve sales comparison",
        "monthly sales vs last year",
        "sales comparison yearly",
        "Compare this month's sales with the same month last year.",

    ],
    "total_shipments": [
        "get monthly total shipments",
        "total num of shipments",
        "retrieve total shipments",
        "calculate monthly shipments",
        "monthly shipments summary",
        "What is the total number of shipments this month?",


    ],
    "revenue_region": [
        "get top revenue sales region",
        "identify highest revenue region",
        "query monthly top sales region",
        "retrieve sales region with highest revenue",
        "monthly sales region leader",
        "Which region generated the most revenue this month?",

 "get least revenue sales region",
        "identify lowest revenue region",
        "query monthly least sales region",
        "retrieve sales region with lowest revenue",
        "monthly sales region leader with lowest revenue",
        "Which region generated the least revenue this month?",

    ],
    
    "sales_by_category": [
        "get total sales by category",
        "query sales by product category",
        "retrieve monthly sales per category",
        "calculate category sales",
        "monthly sales per product category",
        "What is the total sales amount for each product category this month?",

    ],
    "average_order_value": [
        "get monthly average order value",
        "avg order val for this mon",
        "retrieve average order value",
        "calculate monthly order average",
        "monthly order value summary",
        "What’s the average value of orders this month?",

    ]
}


prompts = []
intents = []
for intent, prompt_list in data.items():
    for prompt in prompt_list:
        prompts.append(prompt)
        intents.append(intent)
prompt_embeddings = model.encode(prompts)

import spacy
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from word2number import w2n

# Load spaCy and Sentence Transformer models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined modifier phrases for semantic similarity
modifier_phrases = {
    "most": ["most", "best", "top", "highest", "biggest", "leading"],
    "least": ["least", "worst", "lowest", "smallest", "underperforming"]
}

def detect_modifier_and_number(test_input):
    test_input=test_input.lower()
    user_embedding = model.encode(test_input, convert_to_tensor=True)
    best_modifiers = ["best", "top", "highest", "most"]
    worst_modifiers = ["worst", "least", "lowest", "fewest"]
    intent_embeddings = model.encode(test_input, convert_to_tensor=True)
    best_mod_embeddings = model.encode(best_modifiers, convert_to_tensor=True)
    worst_mod_embeddings = model.encode(worst_modifiers, convert_to_tensor=True)
# Find the intent with the highest similarity
    intent_similarities = util.pytorch_cos_sim(user_embedding, intent_embeddings)
    best_intent_idx = torch.argmax(intent_similarities).item()
    detected_intent = intents[best_intent_idx]

# Detect modifier (best or worst)
    best_mod_similarities = util.pytorch_cos_sim(user_embedding, best_mod_embeddings)
    worst_mod_similarities = util.pytorch_cos_sim(user_embedding, worst_mod_embeddings)

# Get the highest similarity for best and worst
    max_best_mod_similarity = torch.max(best_mod_similarities).item()
    max_worst_mod_similarity = torch.max(worst_mod_similarities).item()

    if max_best_mod_similarity > max_worst_mod_similarity:
        detected_modifier = "most"
    else:
        detected_modifier = "worst"

    return detected_modifier 


from word2number import w2n

def detect_num(test_input):
    number = 1
    tokens = test_input.split()

    
    num_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }

    for token in tokens:
        
        if token.isdigit() and len(token) <= 2:
            number = int(token)
        elif token.lower() in num_words:
            number = num_words[token.lower()]
        else:
            try:
                
                word_number = w2n.word_to_num(token.lower())
                if 0 < word_number <= 99:
                    number = word_number
            except ValueError:
                pass

    return number

from textblob import TextBlob

def correct_spelling(test_input):
    blob = TextBlob(test_input)
    corrected_sentence = blob.correct()
    corrected_prompt = str(corrected_sentence)
    print(corrected_prompt)  
    return corrected_prompt

def detect_intent(corrected_prompt,threshold=0.3):
   
    
    corrected_prompt_embedding = model.encode([corrected_prompt])[0]
    intent_embeddings = model.encode(corrected_prompt, convert_to_tensor=True)
    
    similarities = cosine_similarity([corrected_prompt_embedding], prompt_embeddings)[0]
    
    
    max_similarity_index = np.argmax(similarities)
    max_similarity_score = similarities[max_similarity_index]
    
    
    if max_similarity_score >= threshold:
        return intents[max_similarity_index], max_similarity_score
    else:
        return "Unknown Intent", max_similarity_score

from datetime import datetime, timedelta

def extract_temporal_info(test_input):
    now = datetime.now()
    month_year_info = {
        'month': None,  # Start with None to avoid default values
        'year': now.year,
        'comparison_year': None
    }
    
    lower_text = test_input.lower().strip()
    month_names = {
        "None":0,"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    if "this month" in lower_text:
        month_year_info['month'] = now.month
        month_year_info['year'] = now.year
    elif "last month" in lower_text or "previous month" in lower_text:
        first_of_current_month = now.replace(day=1)
        last_month_end = first_of_current_month - timedelta(days=1)
        month_year_info['month'] = last_month_end.month
        month_year_info['year'] = last_month_end.year
    elif "next month" in lower_text:
        first_of_next_month = (now.replace(day=1) + timedelta(days=31)).replace(day=1)
        month_year_info['month'] = first_of_next_month.month
        month_year_info['year'] = first_of_next_month.year
    elif "this year" in lower_text:
        month_year_info['year'] = now.year
    elif "last year" in lower_text:
        month_year_info['year'] = now.year - 1
    elif "next year" in lower_text:
        month_year_info['year'] = now.year + 1

    words = lower_text.split()
    for i, word in enumerate(words):
        if word in month_names:
            month_year_info['month'] = month_names[word]
            
            if i + 1 < len(words) and words[i + 1].isdigit() and len(words[i + 1]) == 4:
                month_year_info['year'] = int(words[i + 1])
        elif word.isdigit() and len(word) == 4:
            month_year_info['year'] = int(word)

    if "between" in lower_text:
        parts = lower_text.split("between")
        if len(parts) > 1:
            range_part = parts[1].strip().split("and")
            if len(range_part) == 2:
                start_year = range_part[0].strip()
                end_year = range_part[1].strip()
                if start_year.isdigit() and len(start_year) == 4:
                    month_year_info['year'] = int(start_year)
                if end_year.isdigit() and len(end_year) == 4:
                    month_year_info['comparison_year'] = int(end_year)
    elif "for" in lower_text:
        parts = lower_text.split("for")
        if len(parts) > 1:
            range_part = parts[1].strip().split("and")
            if len(range_part) == 2:
                start_year = range_part[0].strip()
                end_year = range_part[1].strip()
                if start_year.isdigit() and len(start_year) == 4:
                    month_year_info['year'] = int(start_year)
                if end_year.isdigit() and len(end_year) == 4:
                    month_year_info['comparison_year'] = int(end_year)
    elif "from" in lower_text:
        parts = lower_text.split("from")
        if len(parts) > 1:
            range_part = parts[1].strip().split("to")
            if len(range_part) == 2:
                start_year = range_part[0].strip()
                end_year = range_part[1].strip()
                if start_year.isdigit() and len(start_year) == 4:
                    month_year_info['year'] = int(start_year)
                if end_year.isdigit() and len(end_year) == 4:
                    month_year_info['comparison_year'] = int(end_year)

    # Ensure comparison_year is set properly if not defined
    if month_year_info['comparison_year'] is None:
        month_year_info['comparison_year'] = month_year_info['year']
    
    return month_year_info


    

def create_query(predicted_intent, modifier, number, month_year_info):
    
    base_query = '''
    WITH BOKHDRData AS (
    SELECT
        bh."BookingHdrAutoId", 
        TO_CHAR(bh."BookingDate", 'MM'::text)::INTEGER AS "BookingMonth",
        TO_CHAR(bh."BookingDate", 'YYYY'::text)::INTEGER AS "BookingYear",
        bh."SalesPersonMasterAutoId",
        bsp."UserName" AS "SalesPersonName",
        cm."CustomerName", 
        bh."CustomerMasterAutoId",
        bh."EbmsTenantAutoId",
        bh."NGStatus",
        lm."LocationName",
        lm."LocationMasterAutoId"
    FROM
        "FF"."BookingHdr" bh
    JOIN "Common"."UserMaster" bsp ON bsp."UserMasterAutoId" = bh."SalesPersonMasterAutoId"
    JOIN "Common"."CustomerMaster" cm ON cm."CustomerMasterAutoId" = bh."CustomerMasterAutoId"
    JOIN "Common"."LocationMaster" lm ON lm."LocationMasterAutoId" = bh."LocationMasterAutoId"
    WHERE
        bh."NGStatus"::text = 'A'::text
),
ACTData AS (
    SELECT
        jh."BookingHdrAutoId", 
        SUM(s1."SellAmount" * s1."ProfitabilityFlag"::numeric) AS "ActualSell"
    FROM
        "Feature"."JobCardSellDtl" s1
    LEFT JOIN "FF"."JobCardHdr" jh ON jh."JobCardHdrAutoId" = s1."FFJobCardHdrAutoId"
    WHERE
        s1."NGStatus"::text = 'A'::text
        AND s1."Status"::text = 'ACTUAL'::text
        AND s1."ChargeSource"::text != 'Consolidation'::text
        AND s1."SellAmount" > 0
        AND jh."BookingHdrAutoId" IS NOT NULL
    GROUP BY
        jh."BookingHdrAutoId"
),
PROData AS (
    SELECT
        bh."BookingHdrAutoId",
        bpdtl."ProductMasterAutoId",
        pm."ProductName"
    FROM
        "FF"."BookingHdr" bh
    JOIN "FF"."BookingPackage" bpdtl ON
        bpdtl."BookingHdrAutoId" = bh."BookingHdrAutoId"
        AND bpdtl."NGStatus"::text = 'A'::text
    JOIN "Common"."ProductMaster" pm ON
        pm."ProductMasterAutoId" = bpdtl."ProductMasterAutoId"
    WHERE
        bh."NGStatus"::text = 'A'::text
)    
SELECT {select_clause}
FROM
    BOKHDRData bhd
JOIN ACTData jcact ON
    jcact."BookingHdrAutoId" = bhd."BookingHdrAutoId"
WHERE
    {where_clause}
    '''
    
    # Determine the where clause based on the presence of month
    if month_year_info['month'] is not None:
        where_clause = f'''
        bhd."BookingMonth" = {month_year_info['month']}
        AND bhd."BookingYear" = {month_year_info['year']}
        OR bhd."BookingYear" BETWEEN {month_year_info['year']} AND {month_year_info['comparison_year'] or month_year_info['year']}
        '''
    else:
        where_clause = f'''
        bhd."BookingYear" = {month_year_info['year']}
        OR bhd."BookingYear" BETWEEN {month_year_info['year']} AND {month_year_info['comparison_year'] or month_year_info['year']}
        '''
    
    query = None
    select_clause = ''
    group_by_clause = ''
    
    if predicted_intent == "sales_performer":
        order_clause = "DESC" if modifier == "most" else "ASC"
        agg = "MAX" if modifier == "most" else "MIN"
        select_clause = f'bhd."SalesPersonName", {agg}(jcact."ActualSell") AS "MaxActRevenue"'
        group_by_clause = 'bhd."SalesPersonName"'
        query = base_query.format(select_clause=select_clause, where_clause=where_clause)
        query += f" GROUP BY {group_by_clause} ORDER BY \"MaxActRevenue\" {order_clause} LIMIT {number};"
    elif predicted_intent == "total_revenue":
        select_clause = 'sum(jcact."ActualSell") AS "totalActRevenue"'
        query = base_query.format(select_clause=select_clause, where_clause=where_clause)
    elif predicted_intent == "sold_product":
        order_clause = "DESC" if modifier == "most" else "ASC"
        select_clause = 'bdata."ProductName", COUNT(bdata."ProductName") AS "Count"'
        group_by_clause = 'bdata."ProductName"'
        query = base_query.format(select_clause=select_clause, where_clause=where_clause)
        query += f" GROUP BY {group_by_clause} ORDER BY \"Count\" {order_clause} LIMIT {number};"
    elif predicted_intent == "revenue_client":
        order_clause = "DESC" if modifier == "most" else "ASC"
        select_clause = 'bhd."CustomerName", sum(jcact."ActualSell") AS "totalActRevenue"'
        group_by_clause = 'bhd."CustomerName"'
        query = base_query.format(select_clause=select_clause, where_clause=where_clause)
        query += f" GROUP BY {group_by_clause} ORDER BY \"totalActRevenue\" {order_clause} LIMIT {number};"
    elif predicted_intent == "sales_comparison":
        year = month_year_info['year']
        comparison_year = month_year_info.get('comparison_year', year - 1)

        select_clause = f'''
        (SUM(CASE WHEN bhd."BookingYear" = {year} THEN jcact."ActualSell" ELSE 0 END) -
         SUM(CASE WHEN bhd."BookingYear" = {comparison_year} THEN jcact."ActualSell" ELSE 0 END))
         AS "RevenueDifference"
        '''

        query = base_query.format(
            select_clause=select_clause,
            where_clause=where_clause
        )

    return query



    


def execute_query(query):
    if query is None:
        print("Error: Query is None. Aborting execution.")
        return None, None
    
    con = db_connection.get_db_connection()
    cur = con.cursor()
    try:
        print(f"Executing query: {query}")
        cur.execute(query)
        results = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        print(f"Query results: {results}")
        cur.close()
        con.close()
        return results, column_names
    except psycopg2.Error as e:
        print("Database error:", e)
        cur.close()
        con.close()
        return None, None


def chatbot_response(test_input):
    if not test_input.strip():
        return "Input is empty. Please provide a question."
    
    # Use correct_spelling function if you want to correct spelling
    corrected_prompt = correct_spelling(test_input)
    
    # Print the corrected prompt for debugging
    print(f"Corrected Prompt: {corrected_prompt}")

    modifier = detect_modifier_and_number(corrected_prompt)
    number = detect_num(corrected_prompt)
    
    predicted_intent, confidence = detect_intent(corrected_prompt)
    
    month_year_info = extract_temporal_info(corrected_prompt)
    
    query = create_query(predicted_intent, modifier, number, month_year_info)
    
    results, column_names = execute_query(query)
    
    if results:
        if predicted_intent == "sales_performer":
            if number == 1:
                response = f"The top salesperson is {results[0][0]} with a revenue of ${results[0][1]:,.2f}."
            else:
                response = "The top sales performers are:"
                for i, row in enumerate(results):
                    response += f"\n{row[0]} with a revenue of ${row[1]:,.2f}"
        elif predicted_intent == "total_revenue":
            response = f"The total sales revenue for the period is ${results[0][0]:,.2f}."
        elif predicted_intent == "sold_product":
            if number == 1:
                response = f"The most sold product is {results[0][0]} with {results[0][1]} units sold."
            else:
                response = "The top-selling products are:"
                for i, row in enumerate(results):
                    response += f"\n{row[0]} with {row[1]} units sold."
        elif predicted_intent == "revenue_client":
            if number == 1:
                response = f"The client with the highest revenue is {results[0][0]} with ${results[0][1]:,.2f} in revenue."
            else:
                response = "The top revenue clients are:"
                for i, row in enumerate(results):
                    response += f"\n{row[0]} with ${row[1]:,.2f} in revenue."
        elif predicted_intent == "sales_comparison":
            response = f"The revenue difference between this month and last year is ${results[0][0]:,.2f}."
        
        return {
        
            "message": response
        }
    else:
        return "No results found or query execution failed."



@app.route('/ask', methods=['POST'])
def ask():
    try:
        test_input = request.json.get("Question")
      
        response = chatbot_response(test_input)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure Flask binds to the correct port
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
