from datetime import datetime, timedelta

LATITUDE = 24.8607
LONGITUDE = 67.0011

END_DATE = datetime.today() - timedelta(days=1)
START_DATE = "2022-08-04"


END_DATE = END_DATE.strftime("%Y-%m-%d")