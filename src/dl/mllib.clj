(ns dl.mllib)

;------------------------------------------------------------------------------------------
(defn matrix [xs]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (map #(list %) xs))

(defn transpose [matrix]
  (apply map vector matrix))

(defn product [a b]
  (doseq [x a] (assert (= (count x) (count b)) "The number of columns in first matrix should be equal to the number of rows in the second"))
  (doseq [x b] (assert (= (count (first b)) (count x)) "Number of columns in second matrix is not consistent"))
  (let [nested-for (fn [f x y] (map (fn [a] (map (fn [b] (f a b)) y)) x))]
    (nested-for (fn [x y] (reduce + (map * x y))) a (transpose b))))

;------------------------------------------------------------------------------------------
(defn layer-calculation [inputs weights activation-fn]
  "forward propagate the input of a layer"
  ; inputs [[1] [0]]
  ; weights [[0.12 0.01] [0.2 0.02] [0.13 0.03]]
  (map #(map activation-fn %)
        (product weights inputs)))

(defn output-errors-calculation [outputs targets dactivation-fn]
  "calculate the errors for the output layer (target value - actual value)"
  ; outputs [[0.02] [0.027]]
  ; targets [0 1]
  (let [outputs (flatten outputs) targets (flatten targets)]
    (map *
         (map dactivation-fn outputs)
         (map - targets outputs))))

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
    (map *
         (map dactivation-fn outputs)
         (flatten (product (matrix errors) (transpose weights))))))

(defn weights-calculation [neurons errors weights rate]
  "update/adjust the weights of a hidden layer according to the errors"
  ; neurons [[0.02] [0.02]]
  ; errors [0.8 0.5]
  ; weights [[0.15 0.02 0.01] [0.16 0.03 0.02]] e.g. 3 neurons follow
  (let [weight-changes (map * errors (flatten neurons))] ; (0.018520152042568426 0.013804030750041782)
    (map (fn [xs w] (map (fn [x] (+ x w)) xs)) weights weight-changes)))