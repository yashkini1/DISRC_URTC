# DISRC_URTC
Deep Intrinsic Surprise-Regularized Control (DISRC): A Biologically Inspired Mechanism for Efficient Deep Q-Learning in Sparse Environments

This repository contains the official implementation, figures, and metrics for the URTC 2025 paper titled:
“Deep Intrinsic Surprise-Regularized Control (DISRC): A Biologically Inspired Mechanism for Efficient Deep Q-Learning in Sparse Environments.”

Authors:

- Yash Kini (kiniyash3@gmail.com)

- Shiv Davay (davayshiv@gmail.com)

- Shreya Polavarapu (shreyapolavarapu9@gmail.com)

Please contact the authors for any questions, issues, or collaboration requests regarding this repository.

DISRC introduces a novel augmentation to Deep Q-Networks (DQNs) by leveraging intrinsic surprise to scale Q-value updates in real-time. By modeling biological plasticity mechanisms—such as neuromodulatory dopamine dynamics—DISRC adjusts learning strength using a deviation-based surprise signal computed from latent encodings of the agent’s internal state.

This surprise signal enables agents to:

  - Prioritize updates during novel experiences (exploration),
  - Converge conservatively as environments become familiar (stability),
  - Outperform standard DQNs in sparse-reward domains.

Experiments were performed in the MiniGrid-DoorKey-8x8 and MiniGrid-LavaCrossingS9N1 environments.

Citation:

Yash Kini, Shiv Davay, Shreya Polavarapu. 
“Deep Intrinsic Surprise-Regularized Control (DISRC): A Biologically Inspired Mechanism for Efficient Deep Q-Learning in Sparse Environments.” 
IEEE Undergraduate Research Technology Conference (URTC), 2025.

Acknowledgments:

This work was conducted under the mentorship of Dr. Maryam Parsa, Hamed Poursiami, and Shay Snyder through the George Mason University ASSIP 2025 Program.
We acknowledge the support of the Farama Foundation for providing the MiniGrid benchmark suite and thank the IEEE URTC organizing committee for the opportunity to submit and present.
