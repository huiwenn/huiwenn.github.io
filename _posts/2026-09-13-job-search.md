---
layout: post
title: "Job search summer 2026"
date: 2026-09-13
tags: misc
---


> Information sharing.

<!--more-->


{: class="table-of-content"}
* TOC
{:toc}



I went through a job search loop this July and August, and here is a summary of it for information sharing. This is by no means a guide because there are many things that I don't know / didn't do well. I'm sure you can do better than me, good luck!

to start off, the mandatory sankey:

![sankey diagram]({{ '/assets/img/jobsearch/sankey.png' | relative_url }})

(companies O and A were, alas, in the leftmost grey bar)


## Applying

The positions I interviewed for are mostly mid-level and senior MLE (or member of technical staff, if they care to use this name) on the topics of agents / agent infra / continual learning / post training, and a few on forecasting and "world models", a few on research.

The companies I applied to are around 50% public companies, 30% D+ stage startups, and 20% earlier startups. Among the 22, 3 were recruiter reach-outs, 7 were referrals from friends, and the rest was just applying online from the company's websites. 


#### on referrals

in my experience this time i find that, unless the referral is strong (e.g. the person has worked with you before and can speak to the process and outcome) and that they know the hiring manager and the exact opening, a referral only gave me slightly better response rate than applying online directly. This might be because I am a more junior and fungible candidate though - my takeaway is ashbyhq works, still apply!


#### on resume

there is this advice circling on the internet that you have to tailor your resume for each application based on the job description. I didn't do this, because (1) a lot of the JDs are generic (on purpose), (2) AI can omit seemingly unrelated skills / experiences that would be good to include, (3) AI can hallucinate untruthful details, and fixing them by hand is time consuming. Obviously you should tailor the resume for the job that you _want_ and include the keywords they'd be looking for, but going with one consistent copy of resume did get me decent responses. 


#### on research

my priorities this time were growth and interesting problems. Other than the obvious contenders of frontier labs and star startups, I've found opportunities at research fractions of larger startups/companies to be interesting: e.g. [perplexity research](https://research.perplexity.ai/),  [decagon labs](https://decagon.ai/blog/introducing-decagon-labs),  [scale labs](https://labs.scale.com/),  apple special projects, etc. I enjoyed talking to them - smart people, cool problems, more structure than neolab startups - at the cost of perhaps less security than product-focused roles.


## Interviews

roughly 4 types of interviews. 

### General Programming

- among my non ML programming rounds, only ~30% were leetcode style questions. My hypothesis is that with coding agents companies can make questions that better simulate the real work, which I think is great. 
- what I mean by that is more practical coding questions, for example:
    - for an agent role: design + implement an in-memory file system
    - for a search role: implement a simple index + citation mechanism
    - for recsys: debug + edit a code repo for better real time recommendations update
- I only had 1 interview that allowed AI coding tools, and the interviewer mostly observed how I worked with claude (haiku) in a big repo. not sure if that 45 minutes were informative to the company. 
    - anecdotally, I know someone who put misleading information in `agents.md`  and tested if a candidate can find out. 

### ML programming

- Again, very tailored for the role.  
- I find these to be hard to prepare for - retrospectively, the best approach would be to know what exactly the team is working on and keep up with recent research on that topic.  
- The 1:1 interviews are more or less [in this style](https://torch-leet.vercel.app/). Among them, I was asked to implement GRPO once, Gaussian Splatting once (to test GPU parallelism skills, which i did not have enough), and [speculative decoding](https://arxiv.org/abs/2211.17192) twice (!)
- I preferred the take home assignments more. They are more time consuming, but you get to put in more thought and show more expertise. It also provides perspective on what the team is working on / what they are looking for.

### System Design

- Open-ended "build a system that does X" design questions
- Agent / RAG questions are quite straightforward 
    - [This doc](https://docs.google.com/document/d/1qjH-xBHtEUEZI7RuHkuRZ4f4iqYd9gAkXTAztjPWvIQ/edit?usp=sharing) has a good overview of all aspects of a agent system. a bit outdated but it helped me to organize my thinking.
    - specific questions i was asked: 
        - chatbot for internal documents (and how to scale to 100k users?)
        - RAG for customer support given policy docs and past interactions (and how to continually improve answers?)
        - memory system for personal agent
        - chatbot for a team working together (multi-user memory / context / interface)
- General ML roles all ask about recommendation systems, with specifics depending on the business they're in.  
    - I am completely new to this interview type. [This video](https://youtu.be/9U48NlbOzCU?si=uwMIDU0Yg5OfC2CX) helped me with pacing, structure, and level of details in my answers; it also felt more realistic than other resources. 

### Behavioral

It was interesting to talk to the +1 and +2s and figure out their priorities. Vibes wise: the startups each has a distinct culture, you can feel it just by talking to folks. The big companies felt more chaotic and overwhelmed, at least compared to a couple years ago. The two questions I usually go for are (1) what are the most challenging / annoying things about working here, and (2) the vision of where the product / research / team is going in near and far future. 

I might be an idealist, but I get the ick when a team's priority is to defend scope and collect paychecks for as long as possible; I want to find a place where we build. 

### What i would've done differently

these are mostly for myself and are subject to further reflection.

- reach out to more people. The two positions I interviewed at apple were not the _best_ fits for my skillset - so i didn't get them - but by telling the stories it turns out that I knew quite some people working on ML at apple, and they pointed me to better fitting teams and projects, plus companies where good managers have gone to start teams. By the time this happened I was already in the offer negotiation stage, so here's a reminder to talk to more people earlier on in the process. 
- Do more thorough prepping for ML coding, for each of the positions. The ML rounds were more in-depth and up-to-date than I expected, and if I had the time / mental capacity I would dig deeper into the field I am going into, read up, implement the common algorithms, and know the trade-offs. 
- work on perspective and signals for corporate america: leadership, ownership, conflict and stress handling, etc. 
