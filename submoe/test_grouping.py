import unittest
import torch
from submoe.grouping import cluster_expert_outputs, tokenwise_cosine_similarity

class SubMoeTest(unittest.TestCase):
    def test_tokenwise_cosine_not_pooled_cosine(self):
        a=torch.tensor([[1.,0.],[0.,1.]])
        b=torch.tensor([[1.,0.],[0.,-1.]])
        self.assertAlmostEqual(float(tokenwise_cosine_similarity(a,b,device='cpu')),0.)
    def test_tokenwise_centroid_and_labels(self):
        reps=torch.tensor([[[1.,0.],[1.,0.]],[[.9,.1],[.9,.1]],[[0.,1.],[0.,1.]],[[.1,.9],[.1,.9]]])
        result=cluster_expert_outputs(reps,2,seed=0,chunk_size=2,device='cpu')
        self.assertEqual(result.labels[0].item(),result.labels[1].item())
        self.assertEqual(result.labels[2].item(),result.labels[3].item())
        self.assertNotEqual(result.labels[0].item(),result.labels[2].item())
        self.assertTrue(result.converged)
if __name__=='__main__': unittest.main()
