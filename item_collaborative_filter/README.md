# Item Collaborative Filtering

This Section explains the pipeline to generate potential candidater for each session to recommend.  
Given we don't have actual attribute(product category... etc) of each aid, items similarities will purely be built from the users' activities.   
With that said, the following characteristics are mainly considered when recalling candidates to recommend.


1. Timestamp adjacency: More relevant if two aid are interacted close to each other on time domain.
2. Order adjacency: More relevant if two aid are interacted close to each other in the order of its session, i.e. 1st and 3rd interaction will be more relevant to 1st and 10th interaction.
3. Action Type: More relevant if the action is more important, i.e. a user made 3 interactions with `item1, item2, item3`, if `item1 `and `item2` are both "buy" action, whereas `item3` is "clicks", `item1 <-> item2` will be more relevant than `item1 <-> item3`
4. User Activity: Item similarity score coming from a very active user are generally less important, according to the IUF("Inverse User Frequency") theories.
5. Item Popularity: Item that are popular(a lot of users interacted with) are generally less relevant, as those tend to be the item ppl will interact with anyways intead of similar interest. 


![image](https://user-images.githubusercontent.com/71299664/218290346-7ce3a938-1e94-4b3e-85a9-90f2ad477d78.png)


