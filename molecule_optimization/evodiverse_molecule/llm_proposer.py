"""Molecular proposals through a user-configured language-model endpoint."""
import re
from rdkit import Chem
from . import crossover as co, mutation as mu
import random
from openai import APIError, OpenAI
MINIMUM = 1e-10


def query_llm(question, temp):
    """Use the user-configured OpenAI-compatible endpoint."""
    import os
    model = os.environ['EVODIVERSE_MODEL']
    with OpenAI(base_url=os.environ['EVODIVERSE_BASE_URL'],
                api_key=os.environ['EVODIVERSE_API_KEY'], timeout=60.0, max_retries=2) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a helpful agent who can optimize molecules based on your knowledge.'},
                {'role': 'user', 'content': question},
            ], temperature=temp)
    return response, response.choices[0].message.content


class LLMMoleculeProposer:
    """Propose molecules with an LLM and fall back to genetic operators."""
    def __init__(self):
        self.task2description = {
                'jnk3': 'I have two molecules and their JNK3 scores (higher is better). The JNK3 score measures the inhibitory ability of a molecule against c-Jun N-terminal kinase 3 (JNK3).\n\n',
                'gsk3b': 'I have two molecules and their GSK3$\beta$ scores (higher is better). The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                }
        self.task2objective = {
                'jnk3': 'Please propose a new child molecule that is likely to improve the jnk3 score.\n\n',
                'gsk3b': 'Please propose a new child molecule that is likely to improve the GSK3$\beta$ score. \n\n',
                }

        self.cold_requirements = """\n\n
        Requirements:
        - Please propose a molecule that has a higher target score based on your knowledge. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.
        - In your response, please provide a concise rationale summarizing how to achieve a higher target score and then propose the molecule. You need to conclude your answer with the sentence below (replacing the placeholder with the SMILES of your proposed molecule):\n\n 
        Based on the above analysis, the proposed molecule is: <box>Molecule SMILES</box>. \n\n """

        self.hot_requirements = """\n\n
        You are participating in a molecular optimization task. Your goal is to explore chemical space while keeping the target score competitive or improved.
        Requirements:
        - Exploration: Propose a molecule that balances novelty (chemical space exploration) and competitiveness (likely high/maintained score).
        - In your response, please provide a rationale summarizing: (i) which space is being explored, (ii) why score should remain competitive. and then propose the molecule. You need to conclude your answer with the sentence below (replacing the placeholder with the SMILES of your proposed molecule):\n\n
        Based on the above analysis, the proposed molecule is: <box>Molecule SMILES</box>.
        \n\n
        """
        self.task=None


    def propose(self, mating_tuples, mutation_rate, temp, query_mode):
        import random, re
        from rdkit import Chem

        task = self.task
        task_definition = self.task2description[task[0]]
        task_objective   = self.task2objective[task[0]]

        parent = [random.choice(mating_tuples), random.choice(mating_tuples)]

        def _unpack(t):
            if len(t) >= 3:
                s, m, w = t[0], t[1], float(t[2])
            else:
                s, m, w = t[0], t[1], 1.0
            return float(s), m, float(w)

        p1_score, p1_mol, p1_w = _unpack(parent[0])
        p2_score, p2_mol, p2_w = _unpack(parent[1])

        parent_mol    = [p1_mol, p2_mol]
        parent_scores = [p1_score, p2_score]
        parent_ws     = [p1_w, p2_w]

        mode = 0 if random.random() < 0.5 else 1

        try:
            p1_smi = Chem.MolToSmiles(p1_mol) if p1_mol is not None else ""
            p2_smi = Chem.MolToSmiles(p2_mol) if p2_mol is not None else ""
        except Exception:
            p1_smi, p2_smi = "", ""

        meta = {
            "parent_weights": parent_ws,  
            "parents": [
                {"smi": p1_smi, "score": p1_score, "weight": p1_w},
                {"smi": p2_smi, "score": p2_score, "weight": p2_w},
            ],
            "mode": mode,
        }

        for retry in range(3):
            try:
                mol_tuple = ""
                for j in range(2):
                    smi_j = Chem.MolToSmiles(parent_mol[j]) if parent_mol[j] is not None else ""
                    tu = "\n[" + smi_j + "," + str(parent_scores[j]) + "]"
                    mol_tuple += tu
                if query_mode == 'cold':
                    prompt = task_definition + mol_tuple + task_objective + self.cold_requirements
                elif query_mode == 'hot':
                    prompt = task_definition + mol_tuple + task_objective + self.hot_requirements

                _, answer = query_llm(prompt, temp=temp)
                proposed_smiles = re.search(r'<box>(.*?)</box>', answer, re.DOTALL).group(1)
                proposed_smiles = sanitize_smiles(proposed_smiles)
                assert proposed_smiles is not None

                new_child = Chem.MolFromSmiles(proposed_smiles)
                print(f"LLM New child SMILES: {Chem.MolToSmiles(new_child)}")

                return new_child, meta
            except APIError:
                # The client already applies bounded retries to transient failures.
                raise
            except Exception as e:
                print(f"{type(e).__name__} {e}")
                print(f"Error in LLM response or SMILES parsing, retrying... ({retry+1}/3)")
                continue

        try:
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            print(f"GA New child SMILES: {Chem.MolToSmiles(new_child)}")
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = parent_mol[0]

        return new_child, meta
    
def sanitize_smiles(smi):
    """Return canonical SMILES, or None when the input is empty or invalid."""
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    except:
        return None
