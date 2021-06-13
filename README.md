# Predicting-Failures-in-HPC-systems

## Summary 

Over the past years, there has been a rapid development in resilience to
HPC (High Performance Computing) by proposing fault tolerant solutions for
applications, reactive recovery approaches such as checkpoint/restart and gaining
a better understanding of system logs. While computing systems have matured
in computing efficiency scale, and evolved architectures, fault manifestation has
become complex. Faults are frequent and are expected to increase in the next
generation systems. Currently, substantial compute capacity and power is wasted
in recovering failed components. Failure prediction with defined lead times and
pin-pointing the node of impending failures is the need of the hour to combat such
unprecedented faults and failing components.
The aim of this repository is firstly to study/visualize logs of IBM’s
supercomputer MIRA and also develop a generic methodology for predicting
failures. This methodology infers failure chains and ultimately tracks the node
ids to pin-point failure location. What’s more using this methodology we can
extract the lead time.
To achieve all of the above, at first logs are being visualized and analyzed
using python3 and pandas. Following, lstms are used to predict failure chains
leading to node failures. A phrase analysis of unlabeled log entries is performed,
which may or may not belong to the failure chain. There is a three-phase deep
learning approach to first train to predict next phrases, second re-train only
sequence of phrases leading to failure chains augmented with expected lead times
and third predict lead times during testing deployment to predict which specific
node fails in how many minutes. 

The methodology of the 3 phases was inspired by Das, Anwesha, et al. "Desh: deep learning for system health prediction of lead times to failure in hpc." Proceedings of the 27th International Symposium on High-Performance Parallel and Distributed Computing. 2018, which was improved by me using transfer learning.

KEYWORDS
Failure Chains, Lead Time, Artificial Intelligence, Deep Learning, Natural
Language Processing, Skip-Gram, LSTM

## Methodology for the 3 phases

#### 1st phase:
Logs were sorted by timestamp.
The phrase of each log was divided by static + dynamic part. The **static** part is conjugated entities of many words, which is encoded to a uniquely identifiable number. The **dynamic** component is discarded.
I use the **skip-gram model** for extracting word embeddings to vectorize the data. Embeddings are defined contexts that check what appears before and after a target event phrase in a sequence of events. By using them I'm creating semantic correlation from vector space models.

##### **Use of Stacked LSTM (2-layers) for training**
* **Input**: a sequence of coded phrases.
* **History Size**: 15.
* **3-step prediction**.
* **Multi-class**.
* **The logs from each node are "connected" to the same LSTM**.

##### **I'm making Phrase Labeling**
* **Safe**: No association with system malfunction. They are eliminated.
* **Error**: A sure sign of an anomaly. Either terminal message or malfunction message.
* **Unknown**: It may or may not be an indication of an anomaly.

I have created a **sequence of events leading to node failure** (Unknown and Error messages).
 The existence of Error terminals is also necessary.
 
### 2nd phase: 
From the 1st phase I have sequences which lead to failure (Having Unknown or Error messages).
I compute ΔΤ (**lead_time**). The message which have ΔΤ=0, are the terminal messages.

##### **Use of Stacked LSTM (2-layers) for training**
* **Input**: ΔΤ and Ids – Multivariate time series.
* **History Size**: 10.
* **1-step prediction**.

In this phase I make use of **Transfer Learning**. With Transfer Learning the starting point of the learning of LSTMs is the model of the 1st phase, this way I achieve faster convergence.

### 3rd phase:
This is the phase of validation of the trained failure chains from phase 2.
The types of phrases that I have in this phase are: Safe, Unknown, Error.
Now the vectors fed to LSTM phase 3 are from a specific node (not concatenated), so logs from each node are fed to the corresponding trained LSTM.

The failure is detected if there is a close match with the real failure chain.
* MSE <= threshold (0.0145) possible failure chain.
* I extract the lead time ΔΤ from the possible failure chains.
