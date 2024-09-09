![preview](https://github.com/user-attachments/assets/42097e44-5413-497b-8a2c-54142e0ad052)
# Robot Utility Models

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.16098-163144.svg?style=for-the-badge)](TODO) -->
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-db6a4b.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)

[Project webpage](https://robotutilitymodels.com) · [Documentation](https://educated-diascia-662.notion.site/Setting-Up-Running-Zero-Shot-Models-on-Hello-Robot-66658ab1a6454f219e0fb1db1baa9d6f) · [Paper](https://robotutilitymodels.com/mfiles/paper/Robot_Utility_Models.pdf)

**Authors**: [Haritheja Etukuru*](https://haritheja.com/), Norihito Naka, [Zijin Hu](https://zij1n.github.io/), [Seungjae Lee](https://sjlee.cc/), [Julian Mehu](https://www.linkedin.com/in/julian-mehu-6aa76725/), [Aaron Edsinger](https://www.linkedin.com/in/aaron-edsinger/), [Chris Paxton](https://cpaxton.github.io/), [Soumith Chintala](https://soumith.ch/), [Lerrel Pinto](https://lerrelpinto.com/), [Nur Muhammad “Mahi” Shafiullah*](https://mahis.life/)

Open-source repository of the hardware and software components of [Robot Utility Models](https://robotutilitymodels.com). 

https://github.com/user-attachments/assets/dc13bcdb-238a-448c-a2ad-27fc15d194f6

<details>
  <summary><h2>Abstract</h2></summary>
  Robot models, particularly those trained with large amounts of data, have recently shown a plethora of real-world manipulation and navigation capabilities. Several independent efforts have shown that given sufficient training data in an environment, robot policies can generalize to demonstrated variations in that environment. However, needing to finetune robot models to every new environment stands in stark contrast to models in language or vision that can be deployed zero-shot for open-world problems. In this work, we present Robot Utility Models (RUMs), a framework for training and deploying zero-shot robot policies that can directly generalize to new environments without any finetuning. To create RUMs efficiently, we develop new tools to quickly collect data for mobile manipulation tasks, integrate such data into a policy with multi-modal imitation learning, and deploy policies on-device on Hello Robot Stretch, a cheap commodity robot, with an external mLLM verifier for retrying. We train five such utility models for opening cabinet doors, opening drawers, picking up napkins, picking up paper bags, and reorienting fallen objects. Our system, on average, achieves 90% success rate in unseen, novel environments interacting with unseen objects. Moreover, the utility models can also succeed in different robot and camera set-ups with no further data, training, or fine-tuning. Primary among our lessons are the importance of training data over training algorithm and policy class, guidance about data scaling, necessity for diverse yet high-quality demonstrations, and a recipe for robot introspection and retrying to improve performance on individual environments.
</details>

## What's on this repo
1. [`hardware`](hardware) contains our 3D printable STL files for the Stick V2, Hello Robot Stretch SE3, and UFactory xArm 7.
3. [`imitation-in-homes`](imitation-in-homes) contains code to download and load one of our robot utility models.
4. [`robot-server`](robot-server) contains code that is run on the robot to deploy the policy.

## Paper
![paper_preview](https://github.com/user-attachments/assets/251bd61f-18a5-4a92-ba01-e524edd3269b)
Get it from our [website](https://robotutilitymodels.com/#paper).


## Citation
If you find any of our work useful, please cite us!
<pre>
@article{etukuru2024robot,
  title={General Policies for Zero-Shot Deployment in New Environments},
  author={Etukuru, Haritheja and Naka, Norihito and Hu, Zijin and Mehu, Julian and Edsinger, Aaron and Paxton, Chris and Chintala, Soumith and Pinto, Lerrel and Shafiullah, Nur Muhammad Mahi},
  year={2024}
}
</pre>