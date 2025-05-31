def route_query(query):
    if "law" in query.lower() or "contract" in query.lower():
        return "legal"
    elif "code" in query.lower() or "function" in query.lower():
        return "code"
    else:
        return "default"
