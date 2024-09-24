# OCR and Keyword Search Web Application

This web application allows users to upload an image containing text in Hindi and English, perform Optical Character Recognition (OCR) on the image, and search for keywords within the extracted text.

## Features

- Image upload
- OCR processing using the General OCR Theory (GOT) model
- Keyword search with highlighted results
- Support for Hindi and English text

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ocr-web-app.git
   cd ocr-web-app
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application Locally

To run the application locally, use the following command:

```
streamlit run app.py
```

The application will open in your default web browser.

## Deployment

This application can be deployed on Streamlit Sharing:

1. Push your code to a GitHub repository.
2. Sign up for Streamlit Sharing (https://share.streamlit.io/).
3. Connect your GitHub account and select the repository containing your app.
4. Streamlit will automatically deploy your app and provide a public URL.

## Usage

1. Upload an image containing Hindi and/or English text.
2. Click the "Perform OCR" button to extract text from the image.
3. Once the OCR is complete, you can enter keywords to search within the extracted text.
4. The search results will be displayed with matching keywords highlighted.

## Dependencies

- streamlit
- torch
- transformers
- pillow

## License

This project is licensed under the MIT License.