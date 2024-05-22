import "bootstrap/dist/css/bootstrap.min.css";
import "./UserInput.css";

function UserInput({ setDescription }) {
  return (
    <textarea
      class="form-control"
      aria-label="With textarea"
      placeholder="Enter your symptoms"
      onChange={(e) => setDescription(e.target.value)}
    ></textarea>
  );
}

export default UserInput;
