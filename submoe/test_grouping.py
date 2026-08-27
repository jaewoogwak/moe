import unittest
import torch
from submoe.grouping import cluster_expert_outputs, tokenwise_cosine_similarity, _similarities

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
    def test_deterministic(self):
        reps=torch.randn(6,5,3)
        self.assertTrue(torch.equal(cluster_expert_outputs(reps,3,seed=9,device='cpu').labels,cluster_expert_outputs(reps,3,seed=9,device='cpu').labels))
    def test_vectorized_matches_reference(self):
        reps=torch.randn(4,7,3); cents=[reps[0],reps[2]]
        actual=_similarities(reps,cents,3,'cpu')
        expected=torch.tensor([[tokenwise_cosine_similarity(r,c,3,'cpu') for c in cents] for r in reps])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
    def test_multiple_empty_recovery(self):
        reps=torch.ones(5,4,2)
        result=cluster_expert_outputs(reps,4,seed=0,device='cpu')
        self.assertEqual(len(torch.unique(result.labels)),4)
        self.assertTrue(result.empty_cluster_events)
        self.assertFalse(torch.isnan(reps[result.labels==0].float().mean(0)).any())
if __name__=='__main__': unittest.main()
