What is Speculative Decoding?

Speculative decoding is a method for speeding up text generation in large language models by having a smaller, faster “draft” model propose several next tokens at once. The larger “target” model then quickly checks whether it agrees with those proposed tokens.

 - If the target model agrees → we accept those tokens without recomputing anything.

 - If it disagrees → we reject the token and fall back to sampling from the target model.

This works because small models are fast, but less accurate, and large models are accurate, but slow.
Speculative decoding lets the large model skip work whenever the small model guessed correctly.