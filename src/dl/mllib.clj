(ns dl.mllib)

;------------------------------------------------------------------------------------------
(defn matrix [xs]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (mapv #(vector %) xs))

(defn transpose [matrix]
  (apply mapv vector matrix))

(defn product [a b]
  (doseq [x a] (assert (= (count x) (count b)) "The number of columns in first matrix should be equal to the number of rows in the second"))
  (doseq [x b] (assert (= (count (first b)) (count x)) "Number of columns in second matrix is not consistent"))
  (let [nested-for (fn [f x y] (mapv (fn [a] (mapv (fn [b] (f a b)) y)) x))]
    (nested-for (fn [x y] (reduce + (mapv * x y))) a (transpose b))))

(defn output-errors-calculation [outputs targets dactivation-fn]
  "calculate the errors for the output layer (target value - actual value)"
  ; outputs [[0.02] [0.027]]
  ; targets [0 1]
  (let [outputs (flatten outputs) targets (flatten targets)]
    (mapv *
         (mapv dactivation-fn outputs)
         (mapv - targets outputs))))

(defn hidden-errors-calculation [outputs errors weights dactivation-fn]
  "calculate errors for the hidden layer based on the output-errors and weights"
  ; errors [0.8 0.5]
  ; weights [[0.15 0.02 0.01] [0.16 0.03 0.02]]
  (println "--- hidden-errors-calculation --B")
  (println "outputs" outputs)
  (println "errors" errors)
  (println "weights" weights)
  (println "--- hidden-errors-calculation --E")
  (let [outputs (flatten outputs)]
    (mapv *
         (mapv dactivation-fn outputs)
         (flatten (product (matrix errors) (transpose weights))))))

(defn weights-calculation [neurons errors weights rate]
  "update/adjust the weights of a hidden layer according to the errors"
  ; neurons [[0.02] [0.02]]
  ; errors [0.8 0.5]
  ; weights [[0.15 0.02 0.01] [0.16 0.03 0.02]] e.g. 3 neurons follow
  (let [weight-changes (mapv * errors (flatten neurons))] ; (0.018520152042568426 0.013804030750041782)
    (mapv (fn [xs w] (mapv (fn [x] (+ x w)) xs)) weights weight-changes)))