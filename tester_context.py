
class Tester:

    def __init__(self, k=[5, 10, 20], seslen = 19):
        self.k = k
        self.session_length = seslen
        self.n_decimals = 4
        self.initialize()

    def initialize(self):
        self.i_count = [0]*self.session_length
        self.recall = [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = [[0]*len(self.k) for i in range(self.session_length)]

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(seq_len):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k)):
                k = self.k[j]
                if target_item in k_predictions.data[:k]:
                    self.recall[i][j] += 1
                    inv_rank = 1.0/self.get_rank(target_item, k_predictions.data[:k])
                    self.mrr[i][j] += inv_rank

            self.i_count[i] += 1


    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])
    
    def format_score_string(self, score_type, score):
        tabs = '\t'
        #if len(score_type) < 8:
        #    tabs += '\t'
        return '\t'+score_type+tabs+score+'\n'

    def get_stats(self):
        score_message = "Recall@5\tMRR@5\tRecall@10\tMRR@10\tRecall@20\tMRR@20\n"
        individual_scores = score_message
        current_recall = [0]*len(self.k)
        current_mrr = [0]*len(self.k)
        current_count = 0
        recall_k = [0]*len(self.k)
        for i in range(self.session_length):
            individual_scores += "\ni<="+str(i)+"\t"
            for j in range(len(self.k)):
                
                r = self.recall[i][j]/self.i_count[i]
                m = self.mrr[i][j]/self.i_count[i]
                
                individual_scores += str(round(r, self.n_decimals))+'\t'
                individual_scores += str(round(m, self.n_decimals))+'\t'

        for i in range(self.session_length):
            score_message += "\ni<="+str(i)+"\t"
            current_count += self.i_count[i]
            for j in range(len(self.k)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                k = self.k[j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                
                score_message += str(round(r, self.n_decimals))+'\t'
                score_message += str(round(m, self.n_decimals))+'\t'

                recall_k[j] = r

        recall5 = recall_k[0]
        recall20 = recall_k[2]

        return individual_scores ,score_message, recall5, recall20

    def get_stats_and_reset(self):
        message = self.get_stats()
        self.initialize()
        return message
