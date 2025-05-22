import training.core as training

def main():
	training.extract_media()
	training.xgb_prep()
	training.xgb_build()
	training.predictions_to_db()

if __name__ == "__main__":
	main()


