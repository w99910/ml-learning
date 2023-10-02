from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# translate Hindi to English
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The head of the UN says there is no military solution in Syria."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
