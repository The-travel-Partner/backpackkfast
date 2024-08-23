import asyncio
import json
import random

import aiohttp.client


class retrieveplace:
    def __init__(self, northeast, southwest):
        self.northeast = northeast
        self.southwest = southwest
        self.resulttourist = {}
        self.resultother = {}

    async def _send_post_request(self, session, url, data):
        async with session.post(url, json=data) as response:
            response.encoding = 'utf-8'
            return await response.json()

    async def touristplace(self, placetype):

        alternate = 0
        url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": "AIzaSyDiRKM906YNP9P9_QY9Uz-pNqdhb9gXnXc",
            "X-Goog-FieldMask": "places.id,places.location,places.rating,places.userRatingCount,places.types,places.photos,places.displayName",
        }

        datas = []
        for i in range(15):
            lat = random.uniform(self.southwest[0], self.northeast[0])
            lng = random.uniform(self.southwest[1], self.northeast[1])

            data = {
                "includedTypes": [placetype],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lng
                        },
                        "radius": 15000.0
                    }
                }
            }
            data1 = {
                "includedTypes": [placetype],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lng
                        },
                        "radius": 15000.0
                    }
                },
                "rankPreference": "DISTANCE"
            }
            if alternate == 0:
                datas.append(data)
                alternate = 1
            else:
                datas.append(data1)
                alternate = 0
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for data in datas:
                task = self._send_post_request(session, url, data)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            for placetourist in results[0]['places']:

                if placetourist.get("photos") is not None and placetourist.get("rating") is not None:
                    name = placetourist['displayName']['text']
                    lat = placetourist['location']['latitude']
                    lng = placetourist['location']['longitude']
                    rating = placetourist['rating']
                    number = placetourist['userRatingCount']
                    place_id = placetourist['id']
                    photos = placetourist['photos']
                    res = {'name': name, 'lat': lat, 'lng': lng, 'rating': rating, 'number': number,
                           'place_id': place_id, 'type': placetype, 'photos': photos}
                    self.resulttourist[f'{name}'] = res
            await session.close()
            return self.resulttourist

    async def other(self, placetype):
        headerselse = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": "AIzaSyDiRKM906YNP9P9_QY9Uz-pNqdhb9gXnXc",
            "X-Goog-FieldMask": "places.id,places.location,places.rating,places.userRatingCount,places.editorialSummary,places.types,places.photos,places.displayName",
        }
        urlelse = "https://places.googleapis.com/v1/places:searchNearby"
        dataselse = []
        for i in range(3):
            lat = random.uniform(self.southwest[0], self.northeast[0])
            lng = random.uniform(self.southwest[1], self.northeast[1])

            dataelse = {
                "includedTypes": [placetype],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lng
                        },
                        "radius": 15000.0
                    }
                }
            }
            dataselse.append(dataelse)
        async with aiohttp.ClientSession(headers=headerselse) as session:
            tasks = []
            for data in dataselse:
                task = self._send_post_request(session, urlelse, data)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            for placeother in results[0]['places']:

                if placeother.get("photos") is not None and placeother.get("rating") is not None:

                    name = placeother['displayName']['text']
                    lat = placeother['location']['latitude']
                    lng = placeother['location']['longitude']
                    rating = placeother['rating']
                    number = placeother['userRatingCount']
                    place_id = placeother['id']
                    photos = placeother['photos']
                    resother = {'name': name, 'lat': lat, 'lng': lng, 'rating': rating, 'number': number,
                           'place_id': place_id, 'type': placetype, 'photos': photos}

                    self.resultother[f'{name}'] = resother
            return self.resultother

    def duplicateremover(self, js):
        res = {}
        dict_obj = json.loads(json.dumps(js))
        res = {k: v for k, v in dict_obj.items() if k not in res}
        return res


class place(retrieveplace):
    def __init__(self, placetypes, northeast, southwest):
        super().__init__(northeast, southwest)
        self.tourist = {}
        self.museum = {}
        self.night_life = {}
        self.religiousplace = {}
        self.zoo = {}
        self.result = {}
        self.types = placetypes

    async def touristattraction(self):
        index = self.types.index("tourist_attraction")
        res = await self.touristplace(self.types[index])
        self.tourist = super().duplicateremover(res)
        return self.tourist

    async def museums(self):
        index = self.types.index("museum")
        res = await super().other(self.types[index])
        self.museum = super().duplicateremover(res)
        return self.museum

    async def nightlife(self):

        nightindex = self.types.index("night_life")
        if nightindex is not None:
            subtype = ['night_club', 'bar']
            for i in subtype:
                res = await super().other(placetype=i)
                self.night_life.update(super().duplicateremover(res))
        return self.night_life

    async def religious(self):
        religiousindex = self.types.index('religious')
        if religiousindex is not None:
            subtype = ['hindu_temple', 'mosque', 'church']
            for i in subtype:
                res = await super().other(placetype=i)
                self.religiousplace.update(super().duplicateremover(res))

        return self.religious

    async def zoos(self):
        index = self.types.index("zoo")
        res = await super().other(self.types[index])
        self.zoo.update(super().duplicateremover(res))
        return self.zoo

    async def getAll(self):
        for place in self.types:
            if place == "tourist_attraction":
                print("hello")
                res = await self.touristattraction()

                self.result['tourist_attraction'] = res
            elif place == "museum":
                res = await self.museums()
                self.result["museum"] = res
            elif place == "night_life":
                res = await self.nightlife()
                self.result["night_life"] = res
            elif place == "hindu_temple" or place == "mosque" or place == "church":
                res = await self.religious()
                self.result[f"{place}"] = res
            elif place == "zoo":
                res = await self.zoos()
                self.result[f"{place}"] = res
        return self.result


