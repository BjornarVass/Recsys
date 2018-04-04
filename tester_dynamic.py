import pickle

class Tester:

    def __init__(self, seqlen = 19, use_day = False, min_time = 0.5, model_info = "dump", temporal = False, k=[5, 10, 20]):
        self.k = k
        self.session_length = seqlen
        self.n_decimals = 4
        self.pickle_path = model_info
        self.use_day = use_day
        self.temporal = temporal
        self.min_time = min_time
        self.log_id = 0
        self.initialize()

    def initialize(self):
        self.i_count = [0]*self.session_length
        self.first_count = 0
        self.recall = [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = [[0]*len(self.k) for i in range(self.session_length)]
        if(self.temporal):
            self.first_recall = [0]*len(self.k)
            self.first_mrr = [0]*len(self.k)

            #temporal testing structures
            self.time_buckets = [self.min_time, 2, 12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276, 300, 348, 396, 444, 500, 501]
            if(self.use_day):
            	for i in range(len(self.time_buckets)):
            		self.time_buckets[i] = self.time_buckets[i]/24
            self.time_count = [0]*len(self.time_buckets)
            self.time_error = [0]*len(self.time_buckets)
            self.time_percent_error = [0]*len(self.time_buckets)


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

    def evaluate_first_item(self, k_predictions, target_item):
        for j in range(len(self.k)):
            k = self.k[j]
            if target_item in k_predictions.data[:k]:
                self.first_recall[j] += 1
                inv_rank = 1.0/self.get_rank(target_item, k_predictions.data[:k])
                self.first_mrr[j] += inv_rank
        self.first_count += 1

    def evaluate_batch_rec(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])

    def evaluate_batch_temporal(self, predictions, targets, sequence_lengths, first_preds, first_targets):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])
            k_predictions = first_preds[batch_index]
            target_item = first_targets[batch_index]
            self.evaluate_first_item(k_predictions, target_item)

    def evaluate_time(self, prediction, target):
        for i in range(len(self.time_buckets)):
            if(target < self.time_buckets[i] or i == len(self.time_buckets)-1):
                self.time_count[i] += 1
                diff = abs(target-prediction)
                self.time_error[i] += diff
                threshold = 0.5
                if(self.use_day):
                    threshold /= 24
                if(target >= threshold):
                    self.time_percent_error[i] += 100*(diff/target)
                break


    def evaluate_batch_time(self, predictions, targets):
        for i in range(len(predictions)):
            prediction = predictions[i]
            target = targets[i]
            self.evaluate_time(prediction, target)

    
    def format_score_string(self, score_type, score):
        tabs = '\t'
        #if len(score_type) < 8:
        #    tabs += '\t'
        return '\t'+score_type+tabs+score+'\n'

    def get_rec_stats(self):
        score_message = "Recall@5\tRecall@10\tRecall@20\tMRR@5\tMRR@10\tMRR@20\n"
        current_recall = [0]*len(self.k)
        current_mrr = [0]*len(self.k)
        current_count = 0
        recall_k = [0]*len(self.k)
        if(self.temporal):
            recall_line = ""
            mrr_line = ""
            score_message += "\nfirst\t"
            for j in range(len(self.k)):
                r = self.first_recall[j]/self.first_count
                m = self.first_mrr[j]/self.first_count
                recall_line += str(round(r, self.n_decimals))+'\t'
                mrr_line += str(round(m, self.n_decimals))+'\t'
            score_message += recall_line + mrr_line
        for i in range(self.session_length):
            recall_line = ""
            mrr_line = ""
            score_message += "\ni<="+str(i)+"\t"
            current_count += self.i_count[i]
            for j in range(len(self.k)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                k = self.k[j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                
                recall_line += str(round(r, self.n_decimals))+'\t'
                mrr_line += str(round(m, self.n_decimals))+'\t'

                recall_k[j] = r
            score_message += recall_line + mrr_line
        return score_message

    def get_idividual_stats(self):
        individual_scores = "Individual scores\n"
        individual_scores += "Recall@5\tRecall@10\tRecall@20\tMRR@5\tMRR@10\tMRR@20\n"
        for i in range(self.session_length):
            recall_line = ""
            mrr_line = ""
            individual_scores += "\ni<="+str(i)+"\t"
            for j in range(len(self.k)):
                
                r = self.recall[i][j]/self.i_count[i]
                m = self.mrr[i][j]/self.i_count[i]
                
                recall_line += str(round(r, self.n_decimals))+'\t'
                mrr_line += str(round(m, self.n_decimals))+'\t'
            individual_scores += recall_line + mrr_line
        return individual_scores

    def get_time_stats(self):
        time_message = "\t\tMAE\tPercent\t"
        cumulative_count = 0
        cumulative_error = 0
        cumulative_percent = 0
        if(self.use_day):
            prefix = "\ndays<="
        else:
            prefix = "\nhours<="

        #add results of individual timebuckets
        for i in range(len(self.time_buckets)):
            time_message += prefix+str(round(self.time_buckets[i],1))+"\t"
            error = self.time_error[i]/max(self.time_count[i],1)
            percent = self.time_percent_error[i]/max(self.time_count[i],1)
            if(i > 0 and i != len(self.time_buckets)-1):
                cumulative_count += self.time_count[i]
                cumulative_percent += self.time_percent_error[i]
                cumulative_error += self.time_error[i]
            time_message += str(round(error, self.n_decimals))+'\t'
            time_message += str(round(percent, self.n_decimals))+'\t'

        #add cummulative scores
        cumulative_count = max(cumulative_count,1)
        time_output = cumulative_error/cumulative_count
        time_message += "\ntotal-last\t" + str(round(cumulative_error/cumulative_count, self.n_decimals))+'\t' + str(round(cumulative_percent/cumulative_count, self.n_decimals))+'\t'
        last = len(self.time_buckets)-1
        cumulative_count += self.time_count[last]
        cumulative_error += self.time_error[last]
        cumulative_percent += self.time_percent_error[last]
        time_message += "\ntotal\t" + str(round(cumulative_error/cumulative_count, self.n_decimals))+'\t' + str(round(cumulative_percent/cumulative_count, self.n_decimals))+'\t'
        
        return time_message

    def get_stats(self, get_time):
        score_message = self.get_rec_stats()
        individual_scores = self.get_idividual_stats()
        #if time results are requested
        if(get_time):
            time_message = self.get_time_stats()
        else:
            time_message = ""
        return score_message, time_message, individual_scores

    def store_stats(self, get_time):
        #recommendation
        rec_dict = {}
        rec_dict["counts"] = self.i_count
        rec_dict["k"] = self.k
        rec_dict["session_length"] = self.session_length
        rec_dict["recall"] = self.recall
        rec_dict["mrr"] = self.mrr
        rec_dict["temporal"] = self.temporal
        if(self.temporal):
            rec_dict["first_count"] = self.first_count
            rec_dict["first_recall"] = self.first_recall
            rec_dict["first_mrr"] = self.first_mrr

        #time prediction
        time_dict = {}
        if(get_time):
            time_dict["mae"] = self.time_error
            time_dict["count"] = self.time_count
            time_dict["buckets"] = self.time_buckets
            time_dict["percent"] = self.time_percent_error

        pickle_dict = {"rec": rec_dict, "time": time_dict}

        #store pickle
        pickle.dump(pickle_dict, open(self.pickle_path + ".pickle", 'wb'))
        return

    def get_stats_and_reset(self, get_time = False, store = False):
        message = self.get_stats(get_time)
        if(store):
            self.store_stats(get_time)
        self.initialize()
        return message
