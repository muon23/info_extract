import os.path
import unittest
import main


class MyTestCase(unittest.TestCase):
    __WORKING_DIR = "../../data"

    def test_download_pdf(self):
        url = "https://stgendev01.blob.core.windows.net/python-test/nw1.pdf"
        main.download_pdf(url, save_to=self.__WORKING_DIR)

        file_name = os.path.basename(url)
        self.assertEqual(file_name, "nw1.pdf")

        downloaded = os.path.exists(os.path.join(self.__WORKING_DIR, file_name))

        self.assertTrue(downloaded)

    def test_pdf2text(self):
        pdf1 = "nw1.pdf"
        pdf_file = os.path.join(self.__WORKING_DIR, pdf1)
        main.pdf2text(pdf_file, self.__WORKING_DIR)

    def test_pdf2text2(self):
        pdf1 = "BOV.pdf"
        pdf_file = os.path.join(self.__WORKING_DIR, pdf1)
        main.pdf2text(pdf_file, self.__WORKING_DIR)

    def test_text2json(self):
        txt1 = "nw1.txt"
        txt_file = os.path.join(self.__WORKING_DIR, txt1)
        json = main.text2json(txt_file)
        print(json)

    def test_text2json2(self):
        txt1 = "BOV.txt"
        txt_file = os.path.join(self.__WORKING_DIR, txt1)
        json = main.text2json(txt_file)
        print(json)


if __name__ == '__main__':
    unittest.main()
