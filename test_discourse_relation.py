import unittest
import discourseRelationExtraction
import WordEmbeddingModel

class ClassExtractor(unittest.TestCase):
    relation = discourseRelationExtraction.DiscourseRelations()
    model = WordEmbeddingModel.GloveModel(WordEmbeddingModel.GLOVE_PATH)
    def test_cosine_similarity(self):
        sent1 = ['I', 'like', 'coding']
        sent2 = ['I', 'like','testing']
        self.assertNotEqual(self.model.getCosinSimilarity(sent1, sent2), 0)
        sent1 = []
        self.assertEqual(self.model.getCosinSimilarity(sent1, sent2), 0)
        sent2 = []
        self.assertEqual(self.model.getCosinSimilarity(sent1, sent2), 0)