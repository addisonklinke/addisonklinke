### Hi there 👋

<!-- TODO better header/banner image -->

```python
from bs4 import BeautifulSoup
import torch, torchaudio, torchtext
from torch import nn


class UltimateMLPipeline(nn.Module):
    """An engineer can dream"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.asr_model = torchaudio.asr.get_sota_model()
        self.ner_model = torchtext.ner.get_sota_model()

    def forward(self, exec_meeting_audio, implementation_url='https://paperswithcode.com'):
        business_needs = self.asr_model(exec_meeting_audio)
        latest_papers = self.ner_model(BeautifulSoup(implementation_url, 'html.parser'))
        similarity_score = torch.cdist(latest_papers, business_needs.use_case, p=2)
        relevant_ml_task = torch.max(similarity_score)
        top_papers = relevant_ml_task.filter(language__isin=business_needs.tech_stack).order_by(business_needs.kpi)
        best_model = top_papers[0].implementation.get_model(pretrained=True)
        return best_model(business_needs.input)


model = UltimateMLPipeline()
torch.save(model.state_dict(), 'UltimateMLPipeline.ckpt')  # Weights available tomorrow ;)
```

[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/addison-klinke-28768b97/)

<!-- TODO Etsy shop badge -->
<!-- TODO tech stack tool badges -->
<!-- TODO organize into sections -->

[![Addison's GitHub stats](https://github-readme-stats.vercel.app/api?username=addisonklinke&count_private=true&theme=onedark)](https://github.com/anuraghazra/github-readme-stats)

[![Addison's Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=addisonklinke&layout=compact&langs_count=3&exclude_repo=misc&theme=onedark)](https://github.com/anuraghazra/github-readme-stats)

[![Addison's StackOverflow](https://github-readme-stackoverflow.vercel.app/?userID=7446465&layout=compact&theme=dark)](https://stackoverflow.com/users/7446465/addison-klinke)


<!--
**addisonklinke/addisonklinke** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
