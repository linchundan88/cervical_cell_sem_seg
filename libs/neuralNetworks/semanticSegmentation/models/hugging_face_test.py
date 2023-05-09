
from transformers import SegformerForSemanticSegmentation

#outputs = model(**inputs)
#logits = outputs.logits

#nvidia/segformer-b2-finetuned-ade-512-512
#nvidia/segformer-b3-finetuned-ade-512-512


SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",  #segformer-b2-finetuned-ade-512-512  b0, b2, b3
            return_dict=False,
            # num_labels=self.num_classes,
            # id2label=self.id2label,
            # label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

print('OK')