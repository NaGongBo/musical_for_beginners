##################################
# Original Code at: https://github.com/catSirup/KorEDA
# removed some augmentation techniques that is unnecessary for our project
##################################


import random


# a word in the list below must be included at the result texts
_important_keywords = ['웃', '배꼽', '재치'
                       '감동', '눈물',
                       '스토리', '내용',
                       '몰입', '빠져', '집중',
                       '무대', '효과',
                       '노래', '넘버',
                       '춤', '댄스', '안무'
                       '연기', '호연', '열연']

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)

        important = False
        for keywords in _important_keywords:
            if word in keywords or keywords in word:
                important = True

        if important:
            new_words.append(word)
            continue

        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def EDA(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.2, p_rd=0.2, num_aug=6):
    words = sentence.split(' ')
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))


    # rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        if len(a_words) == 0:
            continue
        to_append = " ".join(a_words)

        if to_append not in augmented_sentences:
            augmented_sentences.append(to_append)

    # rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        if len(a_words) == 0:
            continue
        to_append = " ".join(a_words)
        if to_append not in augmented_sentences:
            augmented_sentences.append(to_append)

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    if sentence not in augmented_sentences:
        augmented_sentences.append(sentence)

    return augmented_sentences