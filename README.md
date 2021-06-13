# Predicting-Failures-in-HPC-systems

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
