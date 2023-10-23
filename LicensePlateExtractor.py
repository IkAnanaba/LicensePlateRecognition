import uuid


class LicensePlateExtractor:
    def __init__(self):
        self.details = {}

    def __call__(self, license_plate):
        similar_id =  self.is_similar(license_plate)

        if similar_id is None:
            self.details[uuid.uuid4()] = [[]]

    def is_similar(self, plate_number):
        for plate_id in self.details:
            pass


'''
    NOTES
    
        - Needs to take in an image
        - Gives the image an identifier
        - Performs the recognition on the image a few times
        - Finds the best one and saves the best one
            eg - 'ffff-ffff-ffff-ffffffff' : ['ABC-123DE', ['ABD-325HG', 'ATC-123DE', 'CBC-133DE']]


'''
