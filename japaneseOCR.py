from PIL import Image
import streamlit as st
import time


class JapaneseOCR:
    def __init__(self, ocr_model, extracted_images):
        self.reader = ocr_model

        self.detections = {}
        self.labelsNotDetected = []
        self.ADDRESS = extracted_images[0]
        self.COMPANY_NAME = extracted_images[1]
        self.EMAIL = extracted_images[2]
        self.FAX = extracted_images[3]
        self.FULL_NAME = extracted_images[4]
        self.MOBILE = extracted_images[5]
        self.PHONE_NUMBER = extracted_images[6]
        self.POSITION_NAME = extracted_images[7]
        self.URL = extracted_images[8]

    def processAddress(self, addressResult):
        address = []
        for result in addressResult:
            address.append(result[-2])

        address = " ".join(address)
        if address[0] == 'テ':
            address = "〒" + address[1:]
        return address

    def processCompanyName(self, companyNameResult):
        company = []
        for result in companyNameResult:
            company.append(result[-2])

        company = " ".join(company)
        return company

    def processEmail(self, emailResult):
        email = []
        for result in emailResult:
            email.append(result[-2])

        email = " ".join(email)
        return email

    def processFax(self, faxResult):
        fax = []
        for result in faxResult:
            fax.append(result[-2])

        fax = " ".join(fax)
        return fax

    def processFullName(self, fullNameResult):
        fullName = []
        for result in fullNameResult:
            fullName.append(result[-2])

        fullName = " ".join(fullName)
        return fullName

    def processMobile(self, mobileResult):
        mobile = []
        for result in mobileResult:
            mobile.append(result[-2])

        mobile = " ".join(mobile)
        return mobile

    def processPhoneNumber(self, phoneNumberResult):
        phoneNumber = []
        for result in phoneNumberResult:
            phoneNumber.append(result[-2])

        phoneNumber = " ".join(phoneNumber)
        return phoneNumber

    def processPosition(self, positionResult):
        position = []
        for result in positionResult:
            position.append(result[-2])

        position = " ".join(position)
        return position

    def processURL(self, urlResult):
        url = []
        for result in urlResult:
            url.append(result[-2])

        url = " ".join(url)
        return url

    def recognize(self):

        complete_progress = 0
        recognition_progress = st.progress(complete_progress)
        try:
            addressImg = Image.open(self.ADDRESS)
            if addressImg is not None:
                addressResult = self.reader.readtext(addressImg)
                address = self.processAddress(addressResult)
                self.detections['address'] = address
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Address is not detected")

        try:
            companyNameImg = Image.open(self.COMPANY_NAME)
            if companyNameImg is not None:
                companyNameResult = self.reader.readtext(companyNameImg)
                companyName = self.processCompanyName(companyNameResult)
                self.detections['companyName'] = companyName
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Company Name is not detected")

        try:
            emailImg = Image.open(self.EMAIL)
            if emailImg is not None:
                emailResult = self.reader.readtext(emailImg)
                email = self.processEmail(emailResult)
                self.detections['email'] = email
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Email is not detected")

        try:
            faxImg = Image.open(self.FAX)
            if faxImg is not None:
                faxResult = self.reader.readtext(faxImg)
                fax = self.processFax(faxResult)
                self.detections['fax'] = fax
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Fax is not detected")

        try:
            fullNameImg = Image.open(self.FULL_NAME)
            if fullNameImg is not None:
                fullNameResult = self.reader.readtext(fullNameImg)
                fullName = self.processFullName(fullNameResult)
                self.detections['fullName'] = fullName
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Full Name is not detected")

        try:
            mobileImg = Image.open(self.MOBILE)
            if mobileImg is not None:
                mobileResult = self.reader.readtext(mobileImg)
                mobile = self.processMobile(mobileResult)
                self.detections['mobile'] = mobile
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Mobile is not detected")

        try:
            phoneNumberImg = Image.open(self.PHONE_NUMBER)
            if phoneNumberImg is not None:
                phoneNumberResult = self.reader.readtext(phoneNumberImg)
                phoneNumber = self.processPhoneNumber(phoneNumberResult)
                self.detections['phoneNumber'] = phoneNumber
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Phone Number is not detected")

        try:
            positionNameImg = Image.open(self.POSITION_NAME)
            if positionNameImg is not None:
                positionResult = self.reader.readtext(positionNameImg)
                positionName = self.processPosition(positionResult)
                self.detections['positionName'] = positionName
                complete_progress = complete_progress + 11
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("Position is not detected")

        try:
            urlImg = Image.open(self.URL)
            if urlImg is not None:
                urlResult = self.reader.readtext(urlImg)
                url = self.processURL(urlResult)
                self.detections['url'] = url
                complete_progress = complete_progress + 12
                time.sleep(0.1)
                recognition_progress.progress(complete_progress)
        except (Exception,) as e:
            self.labelsNotDetected.append("URL is not detected")
