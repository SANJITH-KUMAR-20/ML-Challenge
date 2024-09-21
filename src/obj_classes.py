from sklearn.preprocessing import OneHotEncoder
import torch
from transformers import (
    DistilBertTokenizer, DistilBertModel, 
    RobertaTokenizer, RobertaModel, 
    AutoTokenizer, AutoModel,
    TrOCRProcessor, VisionEncoderDecoderModel, DonutProcessor
)
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from PIL import Image
import requests
from PIL import Image, ImageOps
from io import BytesIO
from paddleocr import PaddleOCR
import easyocr
import re
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import os


def filter_data_without_images(df : pd.DataFrame):
    to_be_included = []
    for i in range(len(df)):
        img_path = df.loc[i]['image_link'].split("/")[-1]
        img_path = os.path.join("C://Users//ASUS//Desktop//student_resource 3//images",img_path)
        if not os.path.exists(img_path):
            continue
        to_be_included.append(df.loc[i]['image_link'])
    out = df[df['image_link'].isin(to_be_included)]
    return out

def filter_data_for_hdl(df):
    filtered_df = df[df['entity_name'].isin(['depth', 'height', 'width'])]
    return filtered_df

class EmbedColumns:

    def __init__(self, use_pca=False, pca_components=200):
        self.tokenizer = None
        self.model = None
        self.pca = PCA(n_components=pca_components) if use_pca else None
        self.use_pca = use_pca

    def __set_model(self, kind):
        if kind == "DistilBERT":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif kind == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif kind == "MiniLM":
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Model kind '{kind}' not supported")

    def __get_pca(self, x):
        if self.pca is not None:
            return self.pca.transform(x.reshape(1, -1))
        return x

    def __apply_pooling(self, out, strategy='mean'):
        """
        Applies pooling strategies to get a fixed-size embedding.
        'mean' or 'max' pooling supported.
        """
        if strategy == 'mean':
            return out.mean(dim=1)
        elif strategy == 'max':
            return out.max(dim=1).values
        elif strategy == 'cls':
            return out[:, 0, :]
        else:
            raise ValueError(f"Pooling strategy '{strategy}' not supported")

    def fit_pca(self, dataset_embeddings):
        """
        Fit PCA to a larger dataset of embeddings to avoid fitting on single inputs.
        """
        if self.pca:
            self.pca.fit(dataset_embeddings)

    def get_embeddings(self, val, kind="DistilBERT", pooling_strategy='cls'):
        """
        Get embeddings for the input text. Supports optional pooling strategies.
        """
        self.__set_model(kind)
        inputs = self.tokenizer(val, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state

        pooled_output = self.__apply_pooling(hidden_states, strategy=pooling_strategy)

        embedding = pooled_output.squeeze(0).numpy()

        if self.use_pca:
            embedding = self.__get_pca(embedding)

        return embedding

class OCRFeatureExtractor:
    def __init__(self, model_name="TrOCR", pca_components=200):
        self.pca_components = pca_components
        self.model_name = model_name
        
        if model_name == "TrOCR":
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        elif model_name == "Donut":
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        else:
            raise ValueError("Unsupported model name. Use 'TrOCR' or 'Donut'.")
    def get_arch(self):
        return self.model
    
    def extract_features(self, image_path):
        image = Image.open(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            encoder_outputs = self.model.encoder(pixel_values).last_hidden_state
        
        encoder_features = encoder_outputs.squeeze(0).numpy()
        
        return encoder_features

    def extract_features_from_folder(self, folder_path):
        import os
        features_list = []
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if os.path.isfile(image_path):
                features = self.extract_features(image_path)
                features_list.append(features)
        
        return features_list
    

class ImageFeatureExtractor:
    def __init__(self, use_resnet=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_resnet = use_resnet
        
        if use_resnet:
            # Initialize ResNet50 and move it to GPU if available
            self.resnet = models.resnet50(pretrained=True).to(self.device).eval()
            
            # Preprocessing pipeline for ResNet
            self.preprocess_resnet = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet50 input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def extract_features(self, image_path):
        """Extract image features using ResNet and run the model on CUDA if available."""
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess_resnet(img).unsqueeze(0)  # Add batch dimension

        # Move the image tensor to GPU (if available)
        img_tensor = img_tensor.to(self.device)

        # Extract features with ResNet
        with torch.no_grad():
            features = self.resnet(img_tensor)

        # Move features back to CPU and convert to NumPy array
        return features.cpu().numpy()


class PreProcessPipeline:
    def __repr__(self):
        pass
    def __init__(self, data, ocr_model = "Donut",output_file = "output.csv",batch_size = 500,
                 use_group_id  = False,
                 use_resnet = False,
                 image_root_path = "",
                 max_metric_count = 5,
                 kind = "train"):

        self.data = data
        self.output_file = output_file
        self.batch_size = batch_size
        unique_grp_ids = sorted(list(data["group_id"].unique()))

        self.rev_group_id_mappings = {i : grp_id for i, grp_id in enumerate(unique_grp_ids)}
        self.group_id_mappings = {grp_id : i for i, grp_id in enumerate(unique_grp_ids)}
        self.ocr_feature_extrac = OCRFeatureExtractor(ocr_model)
        self.max_metric_count = max_metric_count
        if use_resnet:
            self.resnet_feature_extractor = ImageFeatureExtractor()
        self.image_root_path = image_root_path
        self.kind = kind
        self.s_one_hot_encode = OneHotEncoder()
        self.met_one_hot = OneHotEncoder()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.use_resnet = use_resnet
        self.use_group_id = use_group_id
        self.map_to_unit= {
            # Length
            'cm': 'centimetre', 'mm': 'millimetre', 'm': 'metre', 'in': 'inch', 'ft': 'foot', 'yd': 'yard',
        }
        self.unique_metrics = list(set(self.map_to_unit.values()))
        self.unique_metrics.append("NAN")
        self.__fit_one_hot()
        
# Example usage
    def __get_full_unit_name(self,unit):
        if unit in self.unique_metrics:
            return unit
        out = self.map_to_unit.get(unit.lower(), unit)
        return out

    
    def get_one_hot_encode(self):
        return self.s_one_hot_encode

    def __fit_one_hot(self):
        self.s_one_hot_encode.fit(self.data[['entity_name']])
        self.met_one_hot.fit([[i] for i in self.unique_metrics])

    def extract_value_unit_from_image(self,image_path):
            img = Image.open(image_path)
            gray_img = ImageOps.grayscale(img)
            
            np_image = np.array(gray_img)
            result = self.ocr.ocr(np_image, cls=True)
            extracted_text = " ".join([res[1][0] for res in result[0]])
            pattern = r'(\d+\.?\d*)\s?(centimetre|metre|cm|ft|millimetre|m|foot|yd|mm|yard|in|inch|feet|")'
            matches = re.findall(pattern, extracted_text, re.IGNORECASE)
            if matches:
                return matches
            else:
                return ""
    def get_group_id_mappings(self):
        return self.group_id_mappings, self.rev_group_id_mappings

    def __transform_for_one_record(self,record):
        image_link ,group_id, entity_name, entity_value = None, None, None, None
        if self.kind == "train":
            image_link ,group_id, entity_name, entity_value = record['image_link'], record['group_id'], record['entity_name'], record['entity_value']
        elif self.kind == "test":
            image_link ,group_id, entity_name = record['image_link'], record['group_id'], record['entity_name']
        try:
            img_path = os.path.join(self.image_root_path,image_link.split("/")[-1])
            transformed_img = self.ocr_feature_extrac.extract_features(img_path)
            resnet_features = self.resnet_feature_extractor.extract_features(img_path) if self.use_resnet else None
            transformed_grp = self.group_id_mappings[group_id] if self.use_group_id else None
            transformed_entity_name = self.s_one_hot_encode.transform([[entity_name]]).toarray()
            if entity_value:
                value, unit = entity_value.split(" ")
                value = float(value)
                transformed_unit = self.met_one_hot.transform([[unit]])
                entity_value = (value, transformed_unit)
            
            matches = self.extract_value_unit_from_image(img_path)
            metric_set = []
            if matches:
                if len(matches) > self.max_metric_count:
                    matches = matches[:self.max_metric_count]
                while len(matches) < self.max_metric_count:
                    matches.append(('0', 'NAN'))
                metric_set = []
                for i in range(len(matches)):
                    val , uni = matches[i][0], matches[i][1]

                    uni = uni.strip()
                    mapped_unit = self.__get_full_unit_name(uni)
                    

                    transformed_unit = self.met_one_hot.transform([[mapped_unit]])

                    val = float(val)

                    metric_set.append([val,transformed_unit])
            else:
                with open("log.txt","w") as f:
                    f.write(f"{image_link} \n")
                return
            return resnet_features, transformed_img, transformed_grp, transformed_entity_name, entity_value, metric_set
        except Exception as e:
            with open("logs.txt","w") as f:
                f.write(f"{image_link} {str(e)} \n")
            return None
    
    def preprocess(self):
        processed_data = []
        for idx, record in tqdm(self.data.iterrows()):
            result = self.__transform_for_one_record(record)
            if result:
                processed_data.append(result)
                if (idx + 1) % self.batch_size == 0 or (idx + 1) == len(self.data):
                    self.save_to_disk(processed_data)
                    processed_data = []

    def save_to_disk(self, processed_data):
        df = pd.DataFrame(processed_data, columns=['resnet_features', 'group_id', 'entity_name_onehot', 'metric_set'])
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, mode='w', header=True, index=False)
        else:
            df.to_csv(self.output_file, mode='a', header=False, index=False)
