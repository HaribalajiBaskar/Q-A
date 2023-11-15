import re
import pdfplumber


def extract_text_from_pdf(file_path, question):
    answer = None
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            matches = re.findall(r"{}(.+?)(?:(?=\d+\.)|$)".format(question), page_text, re.DOTALL)
            if matches:
                answer = matches[0].strip()
                break
    return answer
