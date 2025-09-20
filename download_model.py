from transformers import AutoModelForImageClassification, AutoImageProcessor

# HuggingFace checkpoint
checkpoint = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# Download + save
model = AutoModelForImageClassification.from_pretrained(checkpoint)
processor = AutoImageProcessor.from_pretrained(checkpoint)

# Save to your local project folder
save_dir = "./models/mobilenetv2_kaggle"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print("✅ Model and processor saved to", save_dir)
