from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    raw_dir_path = data_ingestion.initiate_data_ingestion()
    print(f"✔️ Data saved in: {raw_dir_path}")
