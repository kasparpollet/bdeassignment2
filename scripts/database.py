import pandas as pd
from dask import dataframe as dd
import dask

from pymongo import MongoClient
from bson.son import SON
from bson.code import Code


class DataBase:
    """
    The DataBase class creates a connections with a mongo database
    And holds methodes to interact with the database
    """
    
    def __init__(self, collection='reviews'):
        print('\nCreating database connection...')
        client = MongoClient("localhost:27017")
        self.db = client["assignment2"]
        self.collection = self.db[collection]

    def get_all(self, collection=None):
        print('\nGetting data...')
        if not collection:
            collection = self.collection
        df = pd.DataFrame(list(collection.find({})))
        df.drop('_id', axis=1, inplace=True)
        return df

    def upload_data(self, df, name, collection=None):
        """
        Upload a given pandas dataframe to the database wth a given table name
        """
        if collection is not None:
            collection = self.collection
        collection.insert_many(df.to_dict(name))
        print('\nSuccessful uploaded data')

    def get_amount_of_reviews_per_hotel(self, hotel_names=None):
        if hotel_names:
            pipeline = [
                {
                    u"$match": {
                        u"Hotel_Name": {
                            u"$in": hotel_names
                        }
                    }
                }, 
                {
                    u"$group": {
                        u"_id": {
                            u"Hotel_Name": u"$Hotel_Name"
                        },
                        u"COUNT(*)": {
                            u"$sum": 1
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Hotel_Name": u"$_id.Hotel_Name",
                        u"COUNT(*)": u"$COUNT(*)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"COUNT(*)", -1) ])
                }
            ]
        else:
            pipeline = [
                {
                    u"$group": {
                        u"_id": {
                            u"Hotel_Name": u"$Hotel_Name"
                        },
                        u"COUNT(*)": {
                            u"$sum": 1
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Hotel_Name": u"$_id.Hotel_Name",
                        u"COUNT(*)": u"$COUNT(*)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"COUNT(*)", -1) ])
                }
            ]

        cursor = self.collection.aggregate(
            pipeline, 
            allowDiskUse = True
        )

        final = {}
        for i in cursor:
            final[i['Hotel_Name']] = i['COUNT(*)']
        return final

    def get_avarage_score_per_hotel(self, hotel_names=None):
        if hotel_names:
            pipeline = [
                {
                    u"$match": {
                        u"Hotel_Name": {
                            u"$in": hotel_names
                        }
                    }
                }, 
                {
                    u"$group": {
                        u"_id": {
                            u"Hotel_Name": u"$Hotel_Name"
                        },
                        u"AVG(Reviewer_Score)": {
                            u"$avg": u"$Reviewer_Score"
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Hotel_Name": u"$_id.Hotel_Name",
                        u"AVG(Reviewer_Score)": u"$AVG(Reviewer_Score)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"AVG(Reviewer_Score)", -1) ])
                }
            ]
        else:
            pipeline = [
                {
                    u"$group": {
                        u"_id": {
                            u"Hotel_Name": u"$Hotel_Name"
                        },
                        u"AVG(Reviewer_Score)": {
                            u"$avg": u"$Reviewer_Score"
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Hotel_Name": u"$_id.Hotel_Name",
                        u"AVG(Reviewer_Score)": u"$AVG(Reviewer_Score)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"AVG(Reviewer_Score)", -1) ])
                }
            ]

        cursor = self.collection.aggregate(
            pipeline, 
            allowDiskUse = True
        )
        
        final = {}
        for i in cursor:
            final[i['Hotel_Name']] = round(i['AVG(Reviewer_Score)'],1)
        return final

    def get_how_many_times_a_score_has_been_given(self, hotel_names=None):
        if hotel_names:
            pipeline = [
                {
                    u"$match": {
                        u"Hotel_Name": {
                            u"$in": hotel_names
                        }
                    }
                }, 
                {
                    u"$group": {
                        u"_id": {
                            u"Reviewer_Score": u"$Reviewer_Score"
                        },
                        u"COUNT(*)": {
                            u"$sum": 1
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Reviewer_Score": u"$_id.Reviewer_Score",
                        u"COUNT(*)": u"$COUNT(*)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"Reviewer_Score", 1) ])
                }
            ]
        else:
            pipeline = [
                {
                    u"$group": {
                        u"_id": {
                            u"Reviewer_Score": u"$Reviewer_Score"
                        },
                        u"COUNT(*)": {
                            u"$sum": 1
                        }
                    }
                }, 
                {
                    u"$project": {
                        u"Reviewer_Score": u"$_id.Reviewer_Score",
                        u"COUNT(*)": u"$COUNT(*)",
                        u"_id": 0
                    }
                }, 
                {
                    u"$sort": SON([ (u"Reviewer_Score", 1) ])
                }            
            ]

        cursor = self.collection.aggregate(
            pipeline,
            allowDiskUse = True
        )

        final = {}
        for i in cursor:
            final[i['Reviewer_Score']] = round(i['COUNT(*)'],1)
        return final

    def get_tags(self, hotel_names=None):

        map = Code('''function() {
                test = this.Tags.replace(/'/g, '"');
                this.Tags = JSON.parse(test);
                this.Tags.forEach((z) => {
                    emit(z, 1);
                });
            }'''
        )
        reduce = Code("function (key, values) {"
               "  var total = 0;"
               "  for (var i = 0; i < values.length; i++) {"
               "    total += values[i];"
               "  }"
               "  return total;"
               "}")
        sort = {
            u"$sort": SON([ (u"value", -1) ])
        }   
        sort = {
            "value": -1
        }   
        if hotel_names:
            query = {
                u"Hotel_Name": {
                    u"$in": hotel_names
                }
            }
        else:
            query = {}

        result = self.db.reviews.map_reduce(map, reduce, "myresults", query=query, sort=sort)

        final = {}
        for i in result.find():
            final[i['_id']] = round(i['value'])

        final2 = {k: v for k, v in sorted(final.items(), key=lambda item: item[1], reverse=True)}
        return final2