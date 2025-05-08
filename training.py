import src.training as training

def main():
	training.extract_media()
	training.xgb_prep()
	training.xgb_build()

if __name__ == "__main__":
	main()


