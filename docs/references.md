# References

Sources the project uses or is likely to cite in the write-up. Other docs cite them by the
key in brackets, for example [Maia]. Each entry says what we use it for.

- **(proposal [n])** means the project proposal already cites it as reference n.
- **Unverified** means the details came from a team doc and haven't been checked against the source. Check them before citing.

Add new sources here rather than in `decisions.md`, and point to them from decisions by key.

## AlphaZero and its predecessors

**[AlphaZero]** D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, D. Hassabis. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science* 362(6419):1140–1144, 2018. https://doi.org/10.1126/science.aar6404
- Our main implementation reference: the chess input planes, the 4,672-slot move encoding, the training setup and the search settings (Dirichlet α = 0.3, 25% noise weight). The supplementary materials include pseudocode.
- The chess input is 119 planes: the current and 7 previous positions, plus rule planes (input-features table in the supplementary materials; checked by Ryan, Nov 20, 2026). We use only the current position plus rule planes; see CLAUDE.md.

**[AlphaZero-arXiv]** D. Silver et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815, 2017. https://arxiv.org/abs/1712.01815
- The preprint of [AlphaZero]. The method reference cites this version. Cite [AlphaZero] in the write-up.

**[AlphaGoZero]** D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, D. Hassabis. "Mastering the game of Go without human knowledge." *Nature* 550(7676):354–359, 2017. Open copy: https://discovery.ucl.ac.uk/id/eprint/10045895/ (proposal [8])
- The residual tower and policy/value heads that AlphaZero reuses.
- The closest prior result to our question (Fig. 3): a supervised network leads early in training, and self-play overtakes it later.

**[AlphaGo]** D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, D. Hassabis. "Mastering the game of Go with deep neural networks and tree search." *Nature* 529(7587):484–489, 2016. https://doi.org/10.1038/nature16961 (proposal [7])
- The original supervised-then-reinforcement-learning pipeline (Fig. 1a). Go only, with policy-gradient RL and no search during training.

## Network components

**[ResNet]** K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition." *CVPR*, 2016, pp. 770–778.
- The residual blocks our network is built from.

**[SE]** J. Hu, L. Shen, G. Sun. "Squeeze-and-Excitation Networks." *CVPR*, 2018, pp. 7132–7141.
- Squeeze-and-excitation blocks, under consideration in decision #14.

**[PUCT]** C. D. Rosin. "Multi-armed bandits with episode context." *Annals of Mathematics and Artificial Intelligence* 61(3):203–230, 2011.
- The origin of the PUCT selection rule that AlphaZero's search uses.

## Human data and reinforcement learning

**[DQfD]** T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, I. Osband, G. Dulac-Arnold, J. Agapiou, J. Z. Leibo, A. Gruslys. "Deep Q-learning from Demonstrations." *AAAI*, 2018. arXiv:1704.03732. https://arxiv.org/abs/1704.03732 (proposal [5], cited there as the 2017 preprint)
- Evidence that the ratio of human demonstrations to self-generated data matters in reinforcement learning.

**[Maia]** R. McIlroy-Young, S. Sen, J. Kleinberg, A. Anderson. "Aligning Superhuman AI with Human Behavior: Chess as a Model System." *KDD*, 2020. https://doi.org/10.1145/3394486.3403219, arXiv:2006.01855
- AlphaZero-style networks trained on Lichess games by rating band. A possible human-like opponent for evaluation.
- Precedent for our low-clock filter (decision #10). Microsoft Research's write-up says Maia discarded moves made with less than 30 seconds left, since those moves are close to random: https://www.microsoft.com/en-us/research/?p=708031. The paper's methods section has the exact wording.

**[CrazyAra]** J. Czech, M. Willig, A. Beyer, K. Kersting, J. Fürnkranz. "Learning to Play the Chess Variant Crazyhouse Above World Champion Level With Deep Neural Networks and Human Data." *Frontiers in Artificial Intelligence* 3:24, 2020. https://doi.org/10.3389/frai.2020.00024, arXiv:1908.06660
- A chess-variant engine trained only with supervised learning on about 570,000 human games, which then beat the crazyhouse world champion. Shows how far human data alone can take an AlphaZero-style network.

**[WarmStart]** H. Wang, M. Preuss, A. Plaat. "Warm-Start AlphaZero Self-Play Search Enhancements." arXiv:2004.12357, 2020. https://arxiv.org/abs/2004.12357 **Unverified.**
- The "cold start" problem of self-play from random weights, and a way to address it without human data.

## AlphaZero at small scale: practical considerations

**[MultiplayerAZ]** N. Petosa, T. Balch. "Multiplayer AlphaZero." arXiv:1910.13012, 2019. https://arxiv.org/abs/1910.13012 (proposal [6])
- A full AlphaZero training loop on modest hardware.

**[HyperSweep]** H. Wang, M. Emmerich, M. Preuss, A. Plaat. "Hyper-Parameter Sweep on AlphaZero General." arXiv:1903.08129, 2019. https://arxiv.org/abs/1903.08129 **Unverified.**
- A method for sweeping one variable at a time and reporting loss, time and Elo, which is close to how we vary the human-data share.

**[ItersEpochs]** H. Wang, M. Emmerich, M. Preuss, A. Plaat. "Analysis of Hyper-Parameters for Small Games: Iterations or Epochs in Self-Play?" arXiv:2003.05988, 2020. https://arxiv.org/abs/2003.05988 **Unverified.**
- On a limited budget, more short iterations beat fewer long ones.

**[KataGo]** D. J. Wu. "Accelerating Self-Play Learning in Go." arXiv:1902.10565, 2019 (AAAI-20 Workshop on Reinforcement Learning in Games). https://arxiv.org/abs/1902.10565
- Self-play improvements that cut compute about 50× compared with earlier AlphaZero-style Go bots. Many of them aren't specific to Go. A good source of efficiency ideas for a compute-limited project.

**[LessonsAZ]** A. Prasad. "Lessons From Implementing AlphaZero" and the "Lessons from AlphaZero" series. Oracle Developers on Medium, 2018. https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191
- A team's practical write-up of building AlphaZero for Connect Four, covering performance optimization and hyperparameter tuning. A blog, not peer-reviewed.

**[NNChess]** D. Klein. "Neural Networks for Chess." arXiv:2209.01506, 2022. https://arxiv.org/abs/2209.01506
- A readable survey of neural chess engines, including Leela Chess Zero's architecture.

**[Lc0]** Leela Chess Zero, an open-source AlphaZero-style chess engine. https://lczero.org, https://github.com/LeelaChessZero/lc0
- Chess-specific refinements to AlphaZero: squeeze-and-excitation blocks, convolutional policy heads, and the FPU reduction in decision #20.

**[minimal_lczero]** M. Carrigan (Rocketknight1). minimal_lczero. https://github.com/Rocketknight1/minimal_lczero **Unverified author.**
- Cited in the method reference for why dropout doesn't help with self-play data.

**[CamUTTT]** Cam's notes from his Ultimate Tic-Tac-Toe AlphaZero project, which trained on the same fish machines. Private, not published.
- The source of decisions #12–#20, the launch script design and the engineering gotchas in CLAUDE.md. Acknowledge it in the write-up rather than citing it.

## Data, tools and reference code

**[Lichess]** Lichess open database. https://database.lichess.org/ (proposal [2])
- Our human games (standard rated games, October 2026), plus puzzles and Stockfish evaluations.

**[Stockfish]** The Stockfish developers. Stockfish 19, released Sep 5, 2026. https://stockfishchess.org/, https://github.com/official-stockfish/Stockfish/releases/tag/sf_19 (proposal [3])
- Evaluation opponent and move-quality judge (decision #21).

**[python-chess]** N. Fiekas. python-chess 1.11.2. https://python-chess.readthedocs.io/ (proposal [1])
- Game rules, PGN parsing, and running Stockfish through `chess.engine`.

**[CPW-AlphaZero]** "AlphaZero." Chess Programming Wiki. https://www.chessprogramming.org/AlphaZero (proposal [4])
- A summary of AlphaZero's chess details with links to follow-up work. The proposal credits it to Vish Abrams and Anthony Young; the wiki has many editors, so check how to credit it.

**[AZGeneral]** S. Nair et al. alpha-zero-general. https://github.com/suragnair/alpha-zero-general
- Reference code to check our MCTS against. Check, don't copy.

**[PettingZoo]** PettingZoo chess environment. https://pettingzoo.farama.org/environments/classic/chess/ **Unverified link.**
- Reference implementation of the AlphaZero move encoding to check ours against.

## Broader "supervised then RL" examples

Listed in the method reference for context if the write-up needs them. **All unverified:** check the details before citing.

- **[AlphaStar]** O. Vinyals et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature* 575:350–354, 2019.
- **[DAPG]** A. Rajeswaran et al. "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations." *RSS*, 2018.
- **[InstructGPT]** L. Ouyang et al. "Training language models to follow instructions with human feedback." *NeurIPS*, 2022.
