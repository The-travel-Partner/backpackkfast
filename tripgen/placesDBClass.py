import copy


class placesDBClass:
    def __init__(self, db, cityname, placetypes):
        self.db = db
        self.cityname = cityname
        self.collection = db['placesdata']
        self.placesdata = {}
        self.placetypes = placetypes

    async def getCity(self):
        res = await self.collection.find_one({"city_name": self.cityname})
        temp= {}
        for types in self.placetypes:
                temp[f'{types}'] = copy.deepcopy(res['places'][types])
        return temp
    async def checkplace(self):
        res = await self.collection.find_one({"city_name": self.cityname})
        return res.keys()
    async def update_json(self, json1 ,json2):
        for key, value in json2.items():
            if isinstance(value, dict):
                # If the value is a nested dictionary and key exists in json1, update it recursively
                json1[key] = await self.update_json(json1.get(key, {}), value)
            else:
                # Otherwise, update or add the value
                json1[key] = value
        return json1
    async def deletePlace(self):
        return "not implemented"

    async def updatePlace(self):
        return "not implemented"
