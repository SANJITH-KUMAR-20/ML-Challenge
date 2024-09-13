
# **ML Challenge: Feature Extraction from Images**

## **Problem Statement**
In this challenge, the goal is to create a machine learning model that extracts **entity values** from product images. This is a crucial capability in various industries like healthcare, e-commerce, and content moderation, where precise product details such as **weight, volume, dimensions, wattage, and voltage** are vital for digital marketplaces.

As digital stores expand, many products lack detailed textual descriptions. Therefore, the ability to extract key information directly from images is essential for enhancing the quality of product listings and customer experience.

## **Data Description**
The dataset includes the following columns:

- **index**: A unique identifier (ID) for each data sample.
- **image_link**: The public URL where the product image is available for download. Example: [sample image link](https://m.media-amazon.com/images/I/71XfHPR36-L.jpg).
  - Images can be downloaded using the `download_images` function from `src/utils.py`. See the **sample code** in `src/test.ipynb`.
- **group_id**: A category code representing the product's group.
- **entity_name**: The name of the entity being extracted from the product image, e.g., `"item_weight"`.
- **entity_value**: The value corresponding to the entity being extracted, e.g., `"34 gram"`.  
  **Note**: For the `test.csv` file, this column will not be present as it's the target variable that your model must predict.

### **Output Format**
Your submission should be in a CSV format with the following columns:

- **index**: The unique identifier (ID) for the data sample (should match the test file index).
- **prediction**: A string in the format `"x unit"`, where:
  - `x` is a float number.
  - `unit` is one of the allowed units listed in the Appendix or `src/constants.py`.
  - Example valid outputs: `"2 gram"`, `"12.5 centimetre"`, `"2.56 ounce"`.
  
**Invalid formats**: `"2 gms"`, `"60 ounce/1.7 kilogram"`, `"2.2e2 kilogram"`, etc.

**Important**:  
- Ensure that every test record in `test.csv` has a corresponding prediction.
- If no entity value is found for a test sample, return an empty string, i.e., `""`.
- Ensure your submission has the exact number of samples as the test file. Mismatch in the number of rows will lead to submission rejection.

### **File Descriptions**
- **Source Files**:
  - `src/sanity.py`: Ensures the final output file passes all formatting checks.
    - Note: This script does not check if there are fewer/more predictions than in the test file.
  - `src/utils.py`: Contains helper functions, such as downloading images from the `image_link`.
  - `src/constants.py`: Contains the allowed units for each entity type.
  - `sample_code.py`: A sample code that generates an output file in the required format. Usage is optional.
  
- **Dataset Files**:
  - `dataset/train.csv`: Training file with labels (`entity_value`).
  - `dataset/test.csv`: Test file without output labels (`entity_value`). Your model should generate predictions for this file.
  - `dataset/sample_test.csv`: Sample test input file.
  - `dataset/sample_test_out.csv`: Sample output file with expected formatting.

### **Constraints**
- Ensure that your submission passes through the **sanity checker**.
  - Your file should receive the message: `Parsing successful for file: ...csv` to confirm correct formatting.
  - Predictions must use the allowed units from `src/constants.py` or the Appendix.
  
### **Evaluation Criteria**
Your submissions will be evaluated based on the **F1 score**, calculated using the following classification logic:

- **True Positives (TP)**: When both the ground truth (GT) and output (OUT) are non-empty, and they match.
- **False Positives (FP)**: When the output is non-empty, but either:
  - It doesn’t match the ground truth, or
  - The ground truth is empty.
- **False Negatives (FN)**: When the output is empty, but the ground truth is non-empty.
- **True Negatives (TN)**: When both the output and ground truth are empty.

**F1 score** =  
`2 * Precision * Recall / (Precision + Recall)`

Where:

- **Precision** = `True Positives / (True Positives + False Positives)`
- **Recall** = `True Positives / (True Positives + False Negatives)`

### **Submission Instructions**
Upload a `test_out.csv` file in the portal with the same format as `sample_test_out.csv`. The file must include the following:

1. **index**: The unique identifier (ID) of each data sample.
2. **prediction**: A string in the format `"x unit"`, adhering to the rules described earlier.

### **Appendix**
Allowed units are provided in the file `src/constants.py` and must be strictly followed. Predictions using other units will be considered **invalid**.

---

By following this structured README, you’ll guide your collaborators or users through the project setup, goals, and constraints, ensuring a clear understanding of the challenge at hand.

