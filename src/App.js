import "bootstrap/dist/css/bootstrap.min.css";
import UserInput from "./User Input/UserInput";
import { useState } from "react";
import "./App.css";

function App() {
  const [description, setDescription] = useState("");
  const [keyWords, setKeyWords] = useState();
  const [predictedDisease, setPredictedDisease] = useState();

  const extract = () => {
    try {
      const response = fetch("http://localhost:5000/extract", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(description),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data.key_words);
          setKeyWords(data.key_words);
        });
    } catch (error) {
      console.error(error);
    }
  };

  const predict = () => {
    var combined = keyWords.join(" ");
    try {
      const response = fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(combined),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data.disease);
          setPredictedDisease(data.disease);
        });
    } catch (error) {
      console.error(error);
    }
  };
  return (
    <div class="main">
      <UserInput setDescription={setDescription} />
      <button type="button" class="btn btn-primary btn-sm" onClick={extract}>
        Extract Key Words
      </button>
      <div id="words">
        <div id="words-extracted">
          <ul class="list-group">
            {keyWords !== undefined &&
              keyWords.map((word) => <li class="list-group-item">{word}</li>)}
          </ul>
        </div>
      </div>
      <button type="button" class="btn btn-primary btn-sm" onClick={predict}>
        Predict Disease
      </button>

      <p>{predictedDisease}</p>
    </div>
  );
}

export default App;
