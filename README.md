# CLIP Directional Prompt Attention for ComfyUI
### What is Directional Prompt Attention?
Direction prompt attention tries to solve the problem of contextual words (or parts of the prompt) having an effect on much later or irrelevant parts of the prompt. For example, this happens often when something is described as a color which makes subsequent parts of the prompt also have this color. [Cutoff for ComfyUI](https://github.com/BlenderNeko/ComfyUI_Cutoff) is script/extension which tries to solve this through isolated prompt masking. However, this can be achieved much easier by simple using an already built-in feature of the CLIP transformer: attention masks. Using attention masks the transformer is limited to only apply attention on certain tokens (words) in the prompt.

A very little known fact about the standard transformer implementation (the one from [Transformers](https://github.com/huggingface/transformers)) is that there is a causal attention mask built in. All commonly used SD models use this CLIP implementation. What this causal attention masking does, is it masks out future tokens from the current tokens attention. What does this mean? Take for example the prompt "a girl had green eyes and red hair", here the token "girl" does not attend to the token "green", but the token "eyes" attend to the token "red". This is a purposeful implementation since transformers are often used in a language modelling setting where they are trained to predict the *next* word, causal attention masks make it so they can not see the future.

However, this may not be desired for use in Stable Diffusion since we want the full prompt to be represented in the outcome image. Using attention masks we can make it so that the token "green" only attends to "eyes" and "red" only attends to "hair". This is what this extension implements.


### How does it work?
Given a prompt, e.g. "a girl had green eyes and red hair", this implementation allows the user to specify a relationship in the prompt using parentheses, `<` and `>`. For example, we can change the prompt to "a (girl < had (green > eyes) and (red > hair))" this makes it so that "green" only applies to "eyes" and "red" only applies to "hair" while the properties of "eyes" and "hair" also only apply to the "girl". Furthermore, this implementation allows to replace the causal attention mask with a full attention mask instead, however, this is very experimental and not the intentional use for which the model was trained.

### ComfyUI nodes
To achieve all of this, the following node is introduced:

**CLIP Directional Prompt Attention Encode:** this node allows the use of `>` and `<` in the prompt to denote relationship between words or parts of the prompt. Note that `<` only works for non-causal attention masks.

You can find this node under `conditioning`

# TODO:
- [ ] Add examples


