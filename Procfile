serving: cd serving && uvicorn api:app --host 0.0.0.0 --port 8080 $PORT
webapp: cd webapp && streamlit run api.py --server.port 8081  $PORT --server.address 0.0.0.0
reporting: cd reporting && python project.py && evidently ui --host 0.0.0.0 --port 8082 $PORT
