


def create_and_save_synonyms(opt, vocab_train, vocab_test, vocab_val):
    # For now save only the base.
    word_embeds = opt.word_embed_path
    dim = opt.word_embed_size
    embed_pth = "{0}_dim{1}_base_synonyms.pickle".format(opt.dataset, dim)
    embed_pth = os.path.join(word_embeds, embed_pth)
    
    if os.path.exists(embed_pth):
        print("Found {}.".format(embed_pth))
        return
    else:
        print("Loading dictionary for synonyms...")
        dictionary=PyDictionary()
        
        # For every v in vocab find synonyms and save to a dict
        synonyms = {}
        all_words = []
        embeds = []
        for v in vocab_train:
            ipdb.set_trace()
            synonyms[v] = v + dictionary.synonym(v.replace(" ", "_")) #wordnet.synsets()
            for syn in synonyms[v]:
                words.extend(syn.split(' '))
                
        pretrained_embedding = Vico(name='linear',
                                    dim=dim,
                                    is_include=lambda w: w in set(words))
        
        dim = 300 if opt.glove else 500
        label_syn_embeds = {}
        for v in vocab_train:
            label_embed = np.zeros(dim)
            for syn in synonyms[v]:
                words = syn.split(' ')
                embed = np.zeros(dim) #TODO
                for w in words:
                    # try
                    embed += pretrained_embedding[w].numpy()
                embed /= len(words)
                label_embed += embed
            label_embed /= len(synonyms[v])
            label_syn_embeds[v] = label_embed
          
        # Pickle the dictionary for later load
        print("Pickling label embeddings averaged over all synonyms including the label itself ...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(label_syn_embeds, f)
        exit(0)
        
if __name__ == "__main__":
    create_and_save_synonyms